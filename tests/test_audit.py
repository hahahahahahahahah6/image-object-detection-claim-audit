import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from image_object_detection_claim_audit import phrases as p
from image_object_detection_claim_audit.audit import evidence, match_phrases, claims, scan, main

class AuditTests(unittest.TestCase):
    def test_phrase_oracle(self):
        self.assertEqual(p.OBJECTDETECTION, ['object detection', 'object-detection', 'objectdetection', 'object detection model', 'object detection models', 'object detection system', 'object detection systems', 'object detection feature', 'object detection capability', 'object detection support', 'supports object detection', 'object detection tool', 'object detection api', 'object detection pipeline', 'object detection algorithm', 'object detection library', 'object detection framework', 'object detection service', 'object detection function', 'object detection engine', 'object detection engines', 'object detector', 'object detectors', 'detects objects', 'detect objects', 'detects object', 'detect object', 'detects objects in images', 'detect objects in images', 'detects objects in photos', 'detect objects in photos', 'detects the object', 'detect the object', 'object detection software', 'object detection app', 'object detection program', 'real-time object detection', 'realtime object detection', 'automatic object detection', 'automated object detection', 'image object detection', 'photo object detection', 'bounding box detection', 'bounding-box detection', 'detects objects in an image', 'detect objects in an image', 'finds objects in images', 'find objects in images', 'object detection in images', 'object detection on images'])
        self.assertEqual(p.NOOBJECTDETECTION, ['does not support object detection', 'do not support object detection', "doesn't support object detection", "don't support object detection", 'no object detection', 'without object detection', 'never performs object detection', 'does not perform object detection', 'object detection not supported', 'object detection unsupported', 'no object detection support', 'lacks object detection', 'missing object detection', 'object detection disabled', 'no object detection capability', 'without object detection support', 'does not provide object detection', 'no object detection model', 'object detection not available', 'does not use object detection', 'no object detector', 'without object detector', 'does not support object detector', 'no object detection feature'])

    def test_generic_oracle(self):
        self.assertEqual(p.GENERIC_NAMES, ['detect_objects', 'detect_object', 'object_detection', 'find_objects', 'locate_objects', 'get_bounding_boxes', 'predict_boxes', 'detect_boxes', 'find_bounding_boxes', 'detect_and_locate', 'run_detection', 'detect_in_image', 'detect_in_frame', 'find_detections', 'get_detections', 'detect_targets', 'detect_objects_in_image', 'object_localization', 'detect_instances', 'predict_detections', 'find_objects_in_image', 'run_object_detection', 'detect_objects_in_frame', 'get_object_boxes'])

    def test_route_oracles(self):
        self.assertEqual((p.YOLO_NAME, p.YOLO_SOURCE), ('YOLO', 'ultralytics'))
        self.assertEqual(p.TORCHVISION_NAMES, ['fasterrcnn_resnet50_fpn', 'retinanet_resnet50_fpn', 'ssd300_vgg16', 'fcos_resnet50_fpn'])
        self.assertEqual(p.PREDICTOR_NAMES, ['DefaultPredictor'])
        self.assertEqual(p.PIPELINE_TASK, 'object-detection')
        self.assertIn('object-detection', 'zero-shot-object-detection')
        self.assertEqual(p.OPENCV_NAMES, ['readNetFromDarknet', 'CascadeClassifier'])
        self.assertEqual(p.OPENCV_METHODS, ['forward', 'detectMultiScale'])

    def test_positive_routes(self):
        constructors = []
        for imp, ctor in [('from ultralytics import YOLO', 'YOLO'), ('from ultralytics import YOLO as Y', 'Y'), ('import ultralytics as u', 'u.YOLO')]:
            for suffix in ['(image)', '.predict(image)']:
                self.assertTrue(evidence(imp+'\nm = '+ctor+'("weights")\nm'+suffix))
        for name in ['fasterrcnn_resnet50_fpn', 'retinanet_resnet50_fpn', 'ssd300_vgg16', 'fcos_resnet50_fpn']:
            constructors += [('from torchvision.models.detection import '+name, name+'()'), ('import torchvision as tv', 'tv.models.detection.'+name+'()')]
        constructors += [('from detectron2.engine import DefaultPredictor', 'DefaultPredictor(cfg)'), ('import detectron2.engine as e', 'e.DefaultPredictor(cfg)')]
        for task in ['"object-detection"', 'task="object-detection"', '"zero-shot-object-detection"']:
            constructors += [('from transformers import pipeline as pipe', 'pipe('+task+')'), ('import transformers as t', 't.pipeline('+task+')')]
        for imp, ctor in constructors:
            for code in ['m = '+ctor+'\nm(image)', ctor+'(image)']:
                with self.subTest(code=code): self.assertTrue(evidence(imp+'\n'+code))
        for name, method in [('readNetFromDarknet','forward'), ('CascadeClassifier','detectMultiScale')]:
            for imp, ctor in [('import cv2', 'cv2.'+('dnn.' if name=='readNetFromDarknet' else '')+name), ('from cv2 import '+name+' as C', 'C')]:
                for code in ['m = '+ctor+'(path)\nm.'+method+'(image)', ctor+'(path).'+method+'(image)']:
                    self.assertTrue(evidence(imp+'\n'+code))
        for name in p.GENERIC_NAMES:
            for code in ['x.'+name+'(image)', 'from anything import '+name+'\n'+name+'(image)', 'from anything import '+name+' as f\nf(image)']:
                self.assertTrue(evidence(code), code)

    def test_rejections(self):
        cases = ['import ultralytics, torchvision, detectron2, transformers, cv2', 'YOLO()(image)', 'other.YOLO()(image)', 'from ultralytics import YOLO\nm=YOLO()']
        for method in ['train', 'val', 'export', 'track']:
            cases.append('from ultralytics import YOLO\nm=YOLO()\nm.'+method+'()')
        for ctor in ['fasterrcnn_resnet50_fpn()', 'DefaultPredictor(cfg)', 'pipeline("object-detection")', 'cv2.dnn.readNetFromDarknet(cfg)', 'cv2.CascadeClassifier(path)']:
            cases.append('from transformers import pipeline\nm='+ctor)
        cases += ['from transformers import pipeline\npipeline("image-classification")(image)', 'from transformers import pipeline\npipeline(model="facebook/detr-resnet-50")(image)']
        cases += [call+'(image)' for call in ['x.classify', 'x.caption', 'x.segment', 'cv2.matchTemplate', 'x.track', 'x.ocr', 'cv2.findContours', 'cv2.HoughCircles']]
        cases += ['"detect_objects(image)" # x.detect_objects()', 'x.detect_objects', 'detect_objects(image)', 'import subprocess\nsubprocess.run(["detector"])', 'this is invalid code !']
        for code in cases:
            with self.subTest(code=code): self.assertFalse(evidence(code))

    def test_shadowing(self):
        routes = [('from ultralytics import YOLO','YOLO','YOLO()(image)'), ('from torchvision.models.detection import fasterrcnn_resnet50_fpn','fasterrcnn_resnet50_fpn','fasterrcnn_resnet50_fpn()(image)'), ('from detectron2.engine import DefaultPredictor','DefaultPredictor','DefaultPredictor()(image)'), ('from transformers import pipeline','pipeline','pipeline("object-detection")(image)'), ('from cv2 import readNetFromDarknet','readNetFromDarknet','readNetFromDarknet().forward()'), ('from cv2 import CascadeClassifier','CascadeClassifier','CascadeClassifier().detectMultiScale()'), ('from x import detect_objects','detect_objects','detect_objects(image)')]
        for imp, name, call in routes:
            for definition in ['def '+name+'(*args): pass', 'class '+name+': pass']:
                self.assertFalse(evidence(imp+'\n'+definition+'\n'+call))

    def test_matching(self):
        self.assertEqual(claims('does not support object detection'), ([], ['does not support object detection']))
        for phrase in ['object detection model', 'object detection models']:
            self.assertEqual(claims(phrase)[0], [phrase])
        self.assertEqual(claims('OBJECT DETECTION')[0], ['object detection'])
        self.assertEqual(claims('bounding box detection tool for labelling training data')[0], ['bounding box detection'])
        self.assertEqual(claims('```py\nobject detection\n```')[0], [])
        self.assertEqual(claims('`object detection`')[0], ['object detection'])
        self.assertEqual(claims('object--detection')[0], [])

    def fixture(self, files, args=()):
        with tempfile.TemporaryDirectory() as tmp:
            for name, content in files.items():
                path=Path(tmp)/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(content)
            out=io.StringIO(); err=io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code=main(['audit', tmp]+list(args))
            return code, out.getvalue(), err.getvalue()

    def test_cli(self):
        rule='documented-image-object-detection-claim-false'
        denial='documented-no-image-object-detection-claim-false'
        self.assertEqual(self.fixture({'README.md':'object detection', 'app.py':'x.detect_objects()'})[0],0)
        code,out,_=self.fixture({'README.md':'OBJECT DETECTION'}, ['--json']); self.assertEqual(code,2)
        finding=json.loads(out)['findings'][0]
        self.assertEqual(finding['rule'],rule); self.assertEqual(finding['severity'],'high')
        self.assertIn('verdict',finding); self.assertIn('message',finding)
        for args, expected in [([],0), (['--fail-on','medium'],2)]:
            self.assertEqual(self.fixture({'tests/test_fixture.py':'"object detection"'},args)[0],expected)
        code,out,_=self.fixture({'README.md':'no object detection','app.py':'x.detect_objects()'}); self.assertEqual(code,2); self.assertIn(denial,out)
        self.assertEqual(self.fixture({'README.md':'hello'})[0],0)
        self.assertEqual(self.fixture({'README.md':'no object detection'})[0],0)
        self.assertEqual(self.fixture({'docs/guide.md':'object detection'},['--ignore','docs/'])[0],0)
        self.assertEqual(self.fixture({'README.md':'object detection','a.py':'from ultralytics import YOLO\nm=YOLO()', 'b.py':'m(image)'})[0],2)
        self.assertEqual(self.fixture({'README.md':'object detection','venv/a.py':'x.detect_objects()'})[0],2)
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'README.md').write_text('hello')
            self.assertEqual(main(['audit','--ignore','docs/',tmp]),0)
        out=io.StringIO()
        with contextlib.redirect_stdout(out): self.assertEqual(main(['--version']),0)
        self.assertEqual(out.getvalue().strip(),'0.1.0')
        self.assertEqual(main(['audit','/nonexistent-claim-audit-path']),1)
        self.assertEqual(main(['--unknown']),1)

    def test_readme_self_audit(self):
        root=Path(__file__).resolve().parents[1]
        self.assertEqual(claims((root/'README.md').read_text()), ([],[]))
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,'README.md').write_text((root/'README.md').read_text())
            self.assertEqual(scan(Path(tmp))[0],[])

    def test_alias_and_shadow_boundaries(self):
        for name in ['fasterrcnn_resnet50_fpn', 'retinanet_resnet50_fpn', 'ssd300_vgg16', 'fcos_resnet50_fpn']:
            self.assertTrue(evidence('from torchvision.models.detection import '+name+' as C\nm=C()\nm(image)'))
            self.assertFalse(evidence('from torchvision.models.detection import '+name+'\ndef '+name+'(): pass\n'+name+'()(image)'))
        for name in p.GENERIC_NAMES:
            self.assertFalse(evidence('from x import '+name+' as f\ndef f(): pass\nf(image)'))
        self.assertFalse(evidence('from ultralytics import YOLO as Y\nclass Y: pass\nY()(image)'))
        self.assertFalse(evidence('import other as ultralytics\nultralytics.YOLO()(image)'))
        self.assertTrue(evidence('import ultralytics\nultralytics.YOLO()(image)'))
        self.assertTrue(evidence('import ultralytics\nultralytics.YOLO().predict(image)'))

    def test_scope_and_collapse(self):
        code,out,_=self.fixture({'README.md':'object detection object detector', 'docs/a.md':'detects objects'})
        self.assertEqual(code,2)
        self.assertEqual(out.count('documented-image-object-detection-claim-false'),1)
        self.assertEqual(self.fixture({'other.md':'object detection'})[0],0)
        self.assertEqual(self.fixture({'docs/deep/guide.md':'object detection'})[0],2)
        self.assertEqual(self.fixture({'README.md':'object detection', 'pkg/model.py':'x.detect_objects()'})[0],0)
        self.assertEqual(self.fixture({'README.md':'object detection', '.git/model.py':'x.detect_objects()'})[0],2)
        self.assertEqual(self.fixture({'README.md':'object detection', 'custom/pyvenv.cfg':'', 'custom/model.py':'x.detect_objects()'})[0],2)
        self.assertEqual(self.fixture({'README.md':'object detection', 'ignored/model.py':'x.detect_objects()'}, ['--ignore','ignored/'])[0],2)
        self.assertEqual(self.fixture({'README.md':'object detection','docs/a.md':'object detection'}, ['--ignore','README','--ignore','docs/'])[0],0)
        code,out,err=self.fixture({'README.md':'hello'}, ['--verbose','--json'])
        self.assertEqual(code,0); self.assertEqual(json.loads(out)['verdict'],'PASS')
        self.assertIn('Scope:',err); self.assertIn('README.md:',err)

    def test_full_checkout_medium(self):
        root=Path(__file__).resolve().parents[1]
        findings,_=scan(root)
        self.assertEqual(len(findings),1)
        self.assertEqual(findings[0]['severity'],'medium')

if __name__ == '__main__': unittest.main()
