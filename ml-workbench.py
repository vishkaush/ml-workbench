import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk as gtk
#import gtk

class MLWB:

  def on_main_window_destroy(self, object, data=None):
    print("Window being closed")
    gtk.main_quit()

  def on_generateResult2Button_clicked(self, object, data=None):
    print("Result2 button clicked")

  def on_generateResult1Button_clicked(self, object, data=None):
    print("Result1 button clicked")

  def on_videoChooserButton_file_set(self, object, data=None):
    print("video file set")

  def on_videoRadio_toggled(self, object, data=None):
    print("video radio toggled")
 
  def on_webCamRadio_toggled(self, object, data=None):
    print("webcam radio toggled")

  def on_model2ComboBox_changed(self, object, data=None):
    print("model2 combobox changed")

  def on_model1ComboBox_changed(self, object, data=None):
    print("model1 combobox changed")

  def on_trainDataFileChooser_file_set(self, object, data=None):
    print("file chooser file set")
  
  def on_taskComboBox_changed(self, object, data=None):
    print("task combobox changed")


  def __init__(self):
    self.gladefile = "workbench-ui.glade"
    self.builder = gtk.Builder()
    self.builder.add_from_file(self.gladefile)
    self.builder.connect_signals(self)
    self.window = self.builder.get_object("main_window")
    self.window.show()

if __name__ == "__main__":
  main = MLWB()
  gtk.main()