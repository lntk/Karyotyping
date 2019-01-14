"""
    @author: lntk
"""
from kivy.core.window import Window, WindowBase
from kivy.graphics.transformation import Matrix
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.scatter import Scatter

from os.path import dirname

from script.gui.popup import ProgressBarPopup, ImagePopup
from script.util import image_utils

image_dir = dirname(__file__) + "/images"


class DraggableImage(BoxLayout):
    def __init__(self, source, **kwargs):
        super().__init__(**kwargs)

        """
        Hierarchy
        """
        self.image = Image(source=source, pos_hint={"center_x": 0.5, "center_y": 0.5})

        self.container = Scatter()
        self.container.add_widget(self.image)

        self.add_widget(self.container)

        """
        Class properties
        """
        self.source = source
        self.scale_value = 1.0
        self.initial_alpha = self.image.color[3]
        self.initial_pos = None
        self.debug = False
        self.clickable = True
        self.touched = False
        self.mouse_pos = tuple()  # mouse position (frequently updated)

        """
        Some updates per drawing
        """
        Window.bind(mouse_pos=lambda w, p: setattr(self, 'mouse_pos', p))
        Window.bind(on_draw=self.fit_image)
        Window.bind(on_dropfile=self._on_file_drop)

    def fit_image(self, *args):
        self.image.size = self.size

    def _on_file_drop(self, window, file_path):
        print("Dropping file ...")
        if self.collide_point(*self.mouse_pos):
            filePath = file_path.decode("utf-8")
            if self.debug:
                print(file_path)
            self.image.source = filePath
            self.image.reload()

            self.source = filePath

    def on_touch_move(self, touch):
        if touch.button == "left" and self.clickable:
            """
            Call scatter method
            """
            self.container.on_touch_move(touch)

            self.clickable = True

            """
            Change image to "moving" state
            """
            self.image.color[3] = self.initial_alpha / 2
            self.image.reload()
        else:
            self.clickable = False

    def on_touch_up(self, touch):
        if self.initial_pos is None:
            self.initial_pos = self.pos

        if touch.button == "left":
            """
            Call scatter method
            """
            self.container.on_touch_up(touch)

            self.clickable = True

            """
            Change image/scatter to normal state
            """
            scale_matrix = Matrix().scale(1.0 / self.container.scale, 1.0 / self.container.scale, 1.0 / self.container.scale)
            self.container.apply_transform(scale_matrix)
            self.image.color[3] = self.initial_alpha
            self.image.reload()

            # TODO: error when resizing window
            self.container.pos = self.initial_pos

    def on_touch_down(self, touch):
        if self.initial_pos is None:
            self.initial_pos = self.pos

        touch.double_tap_time = 1
        if touch.is_double_tap and self.collide_point(*touch.pos):
            self.container.on_touch_down(touch)
            popup = ImagePopup(source=self.image.source)
            popup.open()
            return

        if touch.button == "left" and self.collide_point(*touch.pos):
            self.container.on_touch_down(touch)

            print("Touch ", self)

            """
            Change image to "clicked" state
            """
            scale_matrix = Matrix().scale(self.scale_value, self.scale_value, self.scale_value)
            self.container.apply_transform(scale_matrix)
            self.container.center = self.mouse_pos  # Fix the image to the position of the mouse

    def reload_image(self, source):
        self.image.source = source
        self.image.reload()

    def __eq__(self, other):
        return self is other


class ZoomableImage(BoxLayout):
    def __init__(self, source, input_data=None, **kwargs):
        self.data = input_data
        self.debug = False

        super().__init__(**kwargs)

        self.image = Image(source=source, pos_hint={"center_x": 0.5, "center_y": 0.5})
        self.container = Scatter()
        self.container.add_widget(self.image)

        self.add_widget(self.container)

        # set initial scale
        self.container.scale = 5

        # set initial transparent value
        self.initial_alpha = self.image.color[3]

        # set initial position
        self.initial_pos = None

        self.mouse_pos = tuple()  # mouse position (frequently updated)

        """
        Some updates per drawing
        """
        Window.bind(mouse_pos=lambda w, p: setattr(self, 'mouse_pos', p))
        Window.bind(on_dropfile=self._on_file_drop)

    def on_touch_up(self, touch):
        super(ZoomableImage, self).on_touch_up(touch)
        self.initial_pos = self.container.pos

    def on_touch_down(self, touch):
        super(ZoomableImage, self).on_touch_down(touch)
        if self.initial_pos is None:
            self.initial_pos = self.container.pos

        if touch.is_mouse_scrolling:
            if touch.button == 'scrolldown':
                if self.debug:
                    print('down')
                if self.container.scale < 10:
                    self.container.scale = self.container.scale * 1.1
                    self.container.pos = self.initial_pos

            elif touch.button == 'scrollup':
                if self.debug:
                    print('up')
                if self.container.scale > 1:
                    self.container.scale = self.container.scale * 0.8
                    self.container.pos = self.initial_pos

    def reload_image(self, source):
        self.image.source = source
        self.image.reload()

    def _on_file_drop(self, window, file_path):
        print("Dropping file ...")
        if self.collide_point(*self.mouse_pos):
            filePath = file_path.decode("utf-8")
            if self.debug:
                print(file_path)
            self.image.source = filePath
            self.image.reload()

            self.source = filePath
            if self.data is not None:
                self.data["image"] = image_utils.read_image(filePath)
                print(self.data["image"].shape)


class DroppableImage(BoxLayout):
    def __init__(self, source, **kwargs):
        super().__init__(**kwargs)
        self.image = Image(source=source)
        self.add_widget(self.image)

        # mouse position (frequently updated)
        self.mouse_pos = tuple()
        Window.bind(mouse_pos=lambda w, p: setattr(self, 'mouse_pos', p))

        Window.bind(on_dropfile=self._on_file_drop)

    def _on_file_drop(self, window, file_path):
        # print("Dropping file ...")
        if self.collide_point(*self.mouse_pos):
            filePath = file_path.decode("utf-8")
            self.image.source = filePath
            self.image.reload()


class EmptyImage(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image = Image(source=image_dir + "/question.jpeg")
        self.add_widget(self.image)

        # self.image.size = self.size
