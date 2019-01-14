from kivy.app import App
from kivy.properties import StringProperty
from kivy.lang import Builder

KV = """

#:import win kivy.core.window

<Picture@Scatter>:
   source: None
   on_size: self.center = win.Window.center
   size: image.size
   size_hint: None, None

   Image:
       id: image
       source: root.source

FloatLayout:
   Picture:
       source: ""
   Picture:
       source: ""

"""


class MyApp(App):
    def build(self):
        return Builder.load_string(KV)


MyApp().run()
