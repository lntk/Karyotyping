"""
    @author: lntk
"""
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label


class CenteredButton(Button):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, valign="middle", halign="center")
        self.text_size = (self.width, None)
        self.height = self.texture_size[1]


class ButtonLayout(BoxLayout):
    def __init__(self, buttons, on_presses, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        for name in buttons.keys():
            if name == "empty space":
                empty_space = Label(size_hint=(1.0, buttons[name]))
                self.add_widget(empty_space)
            else:
                button = CenteredButton(text=name, size_hint=(1.0, buttons[name]), on_press=on_presses[name])
                self.add_widget(button)
