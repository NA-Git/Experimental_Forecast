from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
import csv
import time  # Import the time module

class CSVFileLoaderApp(App):
    def build(self):
        self.file_chooser = FileChooserListView()
        self.load_button = Button(text="Load and Process CSV File", size_hint_y=None, height=50)
        self.load_button.bind(on_release=self.load_csv_file)

        self.output_label = Label(text="Select a CSV file to process.", halign="center", size_hint_y=None, height=50)

        self.scroll_view = ScrollView()
        self.data_layout = BoxLayout(orientation="vertical", padding=10, spacing=5)
        self.scroll_view.add_widget(self.data_layout)

        layout = BoxLayout(orientation="vertical", padding=10, spacing=10)
        layout.add_widget(Label(text="Item Auto Attribution Tool", font_size=24, size_hint_y=None, height=50))
        layout.add_widget(self.file_chooser)
        layout.add_widget(self.load_button)
        layout.add_widget(self.output_label)
        layout.add_widget(self.scroll_view)

        return layout

    def process_and_save_csv(self, selected_file):
        try:
            with open(selected_file, 'r') as file:
                csv_reader = csv.reader(file)
                csv_data = list(csv_reader)

            if len(csv_data) > 0:
                # Extract the file name and extension using rsplit
                file_name, file_extension = selected_file.rsplit('.', 1)

                new_file_name = f"{file_name}_new.{file_extension}"

                with open(new_file_name, 'w', newline='') as new_file:
                    csv_writer = csv.writer(new_file)
                    csv_writer.writerows(csv_data)

                #self.display_csv_data(csv_data)
                self.output_label.text = f"File '{new_file_name}' created successfully. The app will now close."
                #time.sleep(10)  # Wait for 2 seconds before closing the app
                #self.stop()  # Close the app
            else:
                self.data_layout.clear_widgets()
                self.output_label.text = "Selected CSV file is empty."
        except Exception as e:
            self.data_layout.clear_widgets()
            self.output_label.text = f"Error: {str(e)}"

    def display_csv_data(self, csv_data):
        self.data_layout.clear_widgets()

        if len(csv_data) > 0:
            for row in csv_data:
                row_layout = BoxLayout(orientation="horizontal", spacing=5)
                for item in row:
                    text_input = TextInput(text=item, readonly=True, multiline=False)
                    row_layout.add_widget(text_input)
                self.data_layout.add_widget(row_layout)

    def load_csv_file(self, instance):
        selected_file = self.file_chooser.selection and self.file_chooser.selection[0] or None

        if selected_file:
            self.process_and_save_csv(selected_file)
        else:
            self.data_layout.clear_widgets()
            self.output_label.text = "No CSV file selected."

if __name__ == "__main__":
    CSVFileLoaderApp().run()