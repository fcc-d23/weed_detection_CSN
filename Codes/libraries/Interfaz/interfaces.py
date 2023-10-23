#LIBRERÍA PARA EL PROCESAMIENTO DE IMÁGENES
#Felipe Calderara Cea (f), Mg Sc PUC, 2021

##
#Importe de librerías externas
import PyQt5
from PyQt5.QtWidgets import QApplication, QLabel, QDialog, QPushButton, QDialogButtonBox, QLineEdit, QFileDialog, QAction, QSpinBox, QDoubleSpinBox, QCheckBox, QVBoxLayout,\
    QFrame, QWidget
#from PyQt5 import uic
from PyQt5.uic import loadUiType
import os, sys
##

code_path = os.path.dirname(os.path.realpath(__file__))
main_lib = code_path[0:0-code_path[::-1].index('\\')]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

##
#Desarrollo de funciones propias

#Carpeta donde están contenidos los diseños gráficos de las UI
ui_folder = code_path + '/UI/'
YNC_prompt_ui = loadUiType(ui_folder + 'YNC_prompt.ui')
class_name_ui = loadUiType(ui_folder + 'class_name.ui')

#Clase para generar interfaces de pregunta Sí, No, Cancelar
class YNC_prompt(YNC_prompt_ui[0], YNC_prompt_ui[1]):
    
    def __init__(self, text):
        
        super().__init__()
        self.setupUi(self)
        self.YNC_button = self.findChild(QDialogButtonBox, 'YNC_button')
        self.prompt = self.findChild(QLabel, 'label')
        self.YNC_button.button(QDialogButtonBox.Yes).clicked.connect(self.accept)
        self.YNC_button.button(QDialogButtonBox.No).clicked.connect(self.reject)
        self.YNC_button.button(QDialogButtonBox.Cancel).clicked.connect(self.finish_exec)
        self.prompt.setText(text)
        
        self.height, self.width = [self.height(), self.width()]
        self.geo_setup()
        
    def finish_exec(self):
            
        self.done(-1)    
        
    def geo_setup(self):
        
        self.setGeometry(50, 50, self.width, self.height)  
        
class class_name(class_name_ui[0], class_name_ui[1]):
    
    def __init__(self):
        
        super().__init__()
        self.setupUi(self)
        self.height, self.width = [self.height(), self.width()]
        self.geo_setup()
        
        self.next_button = self.findChild(QPushButton, 'nextButton')
        self.lineEdit = self.findChild(QLineEdit, 'lineEdit')
        
        self.next_button.clicked.connect(self.name_ok)
        self.show()
    
    def geo_setup(self):
        
        self.setGeometry(50, 50, self.width, self.height)
        
    def name_ok(self):
        
        self.class_name = self.lineEdit.text() 
        self.close()
        return True    
    
    def change_text(self, text):
        
        self.text_line = self.findChild(QLabel, 'class_name_label')
        self.text_line.setText(text)