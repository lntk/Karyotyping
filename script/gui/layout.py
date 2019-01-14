"""
    @author: lntk
"""
from image import DraggableImage, EmptyImage
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.uix.label import Label
from os.path import dirname

image_dir = dirname(__file__) + "/images"


class DroppableLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_image_per_row = 3
        self.images = list()
        # Window.bind(on_draw=None)

    def add_image(self, image):
        self.images.append(image)
        print("Add ", image)
        print(self.images)
        self.rearrange_images()

    def remove_image(self, image):
        self.images.remove(image)
        self.rearrange_images()

    def rearrange_images(self):
        for row in self.children:
            row.clear_widgets()
        self.clear_widgets()

        idx = 0
        num_row = int(len(self.images) / self.num_image_per_row) + 1
        for i in range(num_row):
            row = BoxLayout(orientation="horizontal", spacing=5)
            for j in range(self.num_image_per_row):
                if idx >= len(self.images):
                    break

                # print("Coordinate:", i, j)
                row.add_widget(self.images[i * self.num_image_per_row + j])
                idx += 1

            self.add_widget(row)

            if idx >= len(self.images):
                break


class ChromosomePairLayout(BoxLayout):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.spacing = 10
        self.chromosomes = dict()

        self.chromosome_pairs_layout = BoxLayout(orientation='horizontal', size_hint=(1.0, 0.9), spacing=2)

        # TODO: Modify this for xy-karyotyping image
        if name == "y":
            self.chromosome_1 = EmptyImage()
            self.chromosome_2 = EmptyImage()
        else:
            self.chromosome_1 = DraggableImage(source="")
            self.chromosome_2 = DraggableImage(source="")

        self.chromosomes["1"] = self.chromosome_1
        self.chromosomes["2"] = self.chromosome_2

        self.chromosome_pairs_layout.add_widget(self.chromosome_1)
        self.chromosome_pairs_layout.add_widget(self.chromosome_2)

        name_layout = BoxLayout(orientation='vertical', size_hint=(1.0, 0.1), spacing=10)
        name_layout.add_widget(Image(source="", size_hint=(1.0, 0.5)))
        name_layout.add_widget(Label(text=name, size_hint=(1.0, 0.5)))

        self.add_widget(self.chromosome_pairs_layout)
        self.add_widget(name_layout)

    def get_chromosomes(self):
        return [self.chromosomes["1"], self.chromosomes["2"]]

    def remove_chromosome(self, widget):
        if widget is self.chromosome_1:
            print("Removing 1")
            self.chromosome_pairs_layout.remove_widget(self.chromosome_1)
            self.chromosome_pairs_layout.remove_widget(self.chromosome_2)
            self.chromosome_1 = self.chromosome_2
            self.chromosome_2 = EmptyImage()
            self.chromosome_pairs_layout.add_widget(self.chromosome_1)
            self.chromosome_pairs_layout.add_widget(self.chromosome_2)
            self.chromosomes["1"] = self.chromosome_1
            self.chromosomes["2"] = self.chromosome_2

        elif widget is self.chromosome_2:
            print("Removing 2")
            self.chromosome_pairs_layout.remove_widget(self.chromosome_2)
            self.chromosome_2 = EmptyImage()
            self.chromosome_pairs_layout.add_widget(self.chromosome_2)
            self.chromosomes["2"] = self.chromosome_2

    def is_addable(self, widget):
        if type(widget) is not DraggableImage:
            return False

        if self.collide_point(*widget.container.center):
            if type(self.chromosome_1) is EmptyImage or type(self.chromosome_2) is EmptyImage:
                return True

        return False

    def add_chromosome(self, widget):
        if type(widget) is not DraggableImage:
            return

        if self.collide_point(*widget.container.center):
            if type(self.chromosome_1) is EmptyImage:
                self.chromosome_pairs_layout.remove_widget(self.chromosome_1)
                self.chromosome_pairs_layout.remove_widget(self.chromosome_2)
                self.chromosome_1 = widget
                self.chromosome_2 = EmptyImage()
                self.chromosome_pairs_layout.add_widget(self.chromosome_1)
                self.chromosome_pairs_layout.add_widget(self.chromosome_2)
                self.chromosomes["1"] = self.chromosome_1
                self.chromosomes["2"] = self.chromosome_2

                return
            elif type(self.chromosome_2) is EmptyImage:
                self.chromosome_pairs_layout.remove_widget(self.chromosome_2)
                self.chromosome_2 = widget
                self.chromosome_pairs_layout.add_widget(self.chromosome_2)
                self.chromosomes["2"] = self.chromosome_2

                return

        return

    def load_images(self, source_1, source_2):
        self.chromosome_1.reload_image(source_1)
        self.chromosome_2.reload_image(source_2)
