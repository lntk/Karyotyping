"""
    @author: lntk
"""
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.clock import Clock
import numpy as np

from progressbar import CircularProgressBar, HorizontalProgressBar


class ImagePopup(Popup):
    def __init__(self, source, **kwargs):
        super().__init__(**kwargs)
        self.title = "Image Viewer"
        self.title_align = "center"
        # self.size_hint = (0.5, 0.5)

        self.layout = BoxLayout(orientation="vertical")

        from image import ZoomableImage
        self.image = ZoomableImage(source=source, size_hint=(1.0, 0.9))
        self.button = Button(size_hint=(1.0, 0.1), text="Close")
        self.button.bind(on_press=self.dismiss)

        self.layout.add_widget(self.image)
        self.layout.add_widget(self.button)

        self.content = self.layout


class ProgressBarPopup(Popup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = "Loading"
        self.title_align = "center"
        self.size_hint = (0.5, 0.5)
        # self.auto_dismiss = False

        self.progress_bar = HorizontalProgressBar(size_hint=(1.0, 0.7), pos_hint={"center_x": 0.5, "center_y": 0.5})

        """ Processing layout """
        self.processing_layout = BoxLayout(size_hint=(1.0, 0.3))
        self.processing_info = Label(text="")
        self.processing_layout.add_widget(self.processing_info)

        self.layout = BoxLayout(orientation="vertical")
        self.layout.add_widget(self.progress_bar)
        self.layout.add_widget(self.processing_layout)

        self.content = self.layout

        """ Set up a Clock for loading """
        self.loading_event = Clock.schedule_interval(self.normal_animate, 0.1)

    def set_text(self, text):
        self.processing_info.text = text

    def stop(self):
        self.loading_event.cancel()
        self.loading_event = Clock.schedule_interval(self.speed_up_animate, 0.1)

    def normal_animate(self, dt):
        if self.progress_bar.value < self.progress_bar.max:
            if self.progress_bar.value >= self.progress_bar.max - 1:
                self.progress_bar.set_value(self.progress_bar.max)
            else:
                self.progress_bar.set_value(self.progress_bar.value + 1)
        else:
            self.dismiss()

    def speed_up_animate(self, dt):
        if self.progress_bar.value < self.progress_bar.max:
            if self.progress_bar.value >= self.progress_bar.max - 1:
                self.progress_bar.set_value(self.progress_bar.max)
            else:
                """ Increase the value to a value between the current value and max value """
                new_value = self.progress_bar.value + (self.progress_bar.max - self.progress_bar.value) / 2
                new_value = min(new_value, self.progress_bar.max - 1)
                self.progress_bar.set_value(new_value)
        else:
            self.dismiss()

    def random_animate(self, dt):
        if self.progress_bar.value < self.progress_bar.max:
            if self.progress_bar.value >= self.progress_bar.max - 1:
                self.progress_bar.value = self.progress_bar.max - 1
            else:
                random_ratio = np.random.randint(50) + 1
                new_value = self.progress_bar.value + (self.progress_bar.max - self.progress_bar.value) / random_ratio
                new_value = min(new_value, self.progress_bar.max - 1)
                self.progress_bar.set_value(new_value)
        else:
            self.progress_bar.set_value(0)


class ProgressCirclePopup(Popup):
    pass
