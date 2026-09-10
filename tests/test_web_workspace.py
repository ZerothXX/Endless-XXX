import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from webapp import theme
from webapp.server import create_app, character_name, TOKEN


class ThemeTests(unittest.TestCase):
    def test_alpha_is_respected(self):
        image = Image.new("RGBA", (40, 40), (255, 0, 0, 0))
        image.paste((250, 120, 20, 255), (10, 5, 30, 35))
        result, method = theme.transparent_subject(image)
        self.assertEqual(method, "source_alpha")
        self.assertEqual(result.getpixel((0, 0))[3], 0)

    def test_white_background_does_not_remove_white_interior(self):
        image = Image.new("RGBA", (40, 40), "white")
        image.paste((30, 70, 180), (8, 8, 32, 32))
        image.paste("white", (14, 14, 26, 26))
        result, method = theme.transparent_subject(image)
        self.assertEqual(method, "border_connected_white")
        self.assertEqual(result.getpixel((20, 20))[3], 255)
        self.assertEqual(result.getpixel((0, 0))[3], 0)

    def test_complex_background_requires_segmentation(self):
        result, _ = theme.transparent_subject(Image.new("RGB", (40, 40), "navy"))
        self.assertIsNone(result)

    def test_orange_and_blue_are_distinct_without_character_ids(self):
        outputs=[]
        for color in [(225, 120, 25), (70, 155, 235)]:
            pixels=np.array([color]*700+[(245,245,245)]*200+[(150,70,130)]*100, dtype=np.uint8)
            palette, _ = theme.palette_from_pixels(pixels)
            outputs.append(palette[0]["hex"])
        self.assertNotEqual(*outputs)
        self.assertEqual(outputs[0], "#e17819")

    def test_full_pipeline_cache_and_source_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            folder=root/"dataset"/"orange_test"
            (folder/"images").mkdir(parents=True)
            for n in range(3):
                image=Image.new("RGBA",(50,50),(0,0,255,0))
                image.paste((225,120,25,255),(10,5,40,45))
                image.save(folder/"images"/f"{n}.png")
            (folder/"captions.txt").write_text("0.png, accessory close-up",encoding="utf-8")
            weight=root/"orange_test_final.safetensors"
            weight.write_bytes(b"test fixture, not a real weight")
            with patch.object(theme,"ROOT",root),patch.object(theme,"CACHE",root/"themes"):
                first=theme.build_theme("orange_test",str(weight))
                second=theme.build_theme("orange_test",str(weight))
                self.assertEqual(first,second)
                self.assertEqual(first["palette"][0]["hex"],"#e17819")
                self.assertTrue(first["assets"]["accessory"])
                self.assertEqual(first["provenance"]["accessory_source"],"0.png")
                old=first["fingerprint"]
                (folder/"captions.txt").write_text("0.png, symbol accessory close-up",encoding="utf-8")
                self.assertNotEqual(old,theme.fingerprint(folder,weight))


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client=create_app(start_jobs=False).test_client()
        self.headers={"X-Local-Token":TOKEN}

    def test_index_and_assets(self):
        with self.client.get('/') as response:
            self.assertEqual(response.status_code,200)
            html=response.get_data(as_text=True)
            self.assertIn('<title>无尽的xxx · 角色工作室</title>',html)
            cover=html.split('<template',1)[0]
            self.assertIn('cover-particles',cover)
            self.assertNotIn('cover-pm',cover)
            self.assertNotIn('无尽的可能性',cover)
        with self.client.get('/static/theme.js') as response:
            self.assertEqual(response.status_code,200)

    def test_csrf_and_host(self):
        self.assertEqual(self.client.post('/api/generate').status_code,403)
        self.assertEqual(self.client.get('/api/bootstrap',headers={"Host":"evil.test"}).status_code,403)
        self.assertEqual(self.client.post('/api/generate',headers={**self.headers,"Origin":"https://evil.test"}).status_code,403)

    def test_names_cannot_escape_workspace(self):
        for name in ('../37','a/b','a\\b','..','',None):
            with self.assertRaises(ValueError):character_name(name)
        self.assertEqual(character_name('my_character-2'),'my_character-2')

    def test_model_required_and_no_arbitrary_file_serving(self):
        self.assertEqual(self.client.post('/api/generate',data={'character':'missing_test'},headers=self.headers).status_code,400)
        self.assertEqual(self.client.get('/runs/anything/spec.json').status_code,404)
        self.assertEqual(self.client.get('/themes/../../config.py').status_code,404)

    def test_existing_character_not_overwritten(self):
        buffer=io.BytesIO();Image.new('RGB',(20,20),'orange').save(buffer,format='PNG');buffer.seek(0)
        response=self.client.post('/api/train',data={'name':'37','images':(buffer,'image.png')},headers=self.headers)
        self.assertEqual(response.status_code,400)


if __name__=='__main__': unittest.main()
