import tone_mapping_show
import sys

app = tone_mapping_show.QApplication(sys.argv)
ex = tone_mapping_show.HDRGANApp()
ex.show()
sys.exit(app.exec_())