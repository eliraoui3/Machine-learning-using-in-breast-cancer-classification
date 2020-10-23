# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 11:28:40 2020

@author: AMEL
"""

from PyQt5.QtWidgets import (QMainWindow, QTextEdit,QAction, QFileDialog,QPushButton, QApplication,QLabel,QLineEdit)
from PyQt5.QtGui import QIcon,QPixmap
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
class Example(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setGeometry(5, 35, 1450, 700)
        self.qlabel1=QLabel(self)
        self.qlabel1.setGeometry(20,40,400,40)
        self.qlabel1.setStyleSheet("font: bold 18px;color: black");
        self.qlabel1.setText("ALGORITHMS DE CLASSIFICATION?")
        self.button1 = QPushButton("ARBRE DE DECISION ",self)
        self.button1.setStyleSheet("background-color :violet; border-style: outset;border-width: 2px;border-radius: 10px;border-color: beige;font: bold 14px;min-width: 10em;padding: 6px")
        self.button1.setGeometry(20,80,300,40)
        self.button1.clicked.connect(self.DTreeFunction)
        self.button2 = QPushButton("SUPPORT VECTOR MACHINE ",self)
        self.button2.setStyleSheet("background-color :violet; border-style: outset;border-width: 2px;border-radius: 10px;border-color: beige;font: bold 14px;min-width: 10em;padding: 6px")
        self.button2.setGeometry(20,150,300,40)
        self.button2.clicked.connect(self.SVMFunction)
        self.button3 = QPushButton("K NEAREST NEIGHBOURS ",self)
        self.button3.setStyleSheet("background-color :violet; border-style: outset;border-width: 2px;border-radius: 10px;border-color: beige;font: bold 14px;min-width: 10em;padding: 6px")
        self.button3.setGeometry(20,220,300,40)
        self.button3.clicked.connect(self.knnFunction)
        self.button4 = QPushButton("Clear ALL ",self)
        self.button4.setStyleSheet("background-color :violet; border-style: outset;border-width: 2px;border-radius: 10px;border-color: beige;font: bold 14px;min-width: 10em;padding: 6px")
        self.button4.setGeometry(10,600,10,40)
        self.button4.clicked.connect(self.ClearFunction)
        self.affichage1=QLabel(self)
        self.affichage1.setGeometry(500,20,400,200)
        self.affichage1.setStyleSheet("font: bold 14px;color:black");
        self.affichage2=QLabel(self)
        self.affichage2.setGeometry(980,20,400,200)
        self.affichage2.setStyleSheet("font: bold 14px;color:black");
        self.affichage3=QLabel(self)
        self.affichage3.setGeometry(700,200,500,380)
        self.affichage3.setStyleSheet("font: bold 14px;color:black");
        self.qlabel1=QLabel(self)
        self.qlabel1.setGeometry(20,350,400,40)
        self.qlabel1.setStyleSheet("font: bold 18px;color: black");
        self.qlabel1.setText("Saisie Par Main de Donnée Désirer à classer?")
        self.line1 = QLineEdit(self)
        self.line1.setGeometry(20,400,80,40)
        self.line1.setPlaceholderText("mean_radius ...")
        self.line1.setStyleSheet("font: bold 14px;min-width:10em;border: 2px solid gray;border-radius: 10px;border-radius: 10px;padding: 0 8px;background: LemonChiffon;selection-background-color: darkgray;padding: 6px")
        self.line2 = QLineEdit(self)
        self.line2.setGeometry(250,400,80,40)
        self.line2.setPlaceholderText("mean_texture ...")
        self.line2.setStyleSheet("font: bold 14px;min-width:10em;border: 2px solid gray;border-radius: 10px;border-radius: 10px;padding: 0 8px;background: LemonChiffon;selection-background-color: darkgray;padding: 6px")
        self.line3 = QLineEdit(self)
        self.line3.setGeometry(20,500,80,40)
        self.line3.setPlaceholderText("mean_perimeter ...")
        self.line3.setStyleSheet("font: bold 14px;min-width:10em;border: 2px solid gray;border-radius: 10px;border-radius: 10px;padding: 0 8px;background: LemonChiffon;selection-background-color: darkgray;padding: 6px")
        self.line4 = QLineEdit(self)
        self.line4.setGeometry(250,500,80,40)
        self.line4.setPlaceholderText("mean_area ...")
        self.line4.setStyleSheet("font: bold 14px;min-width:10em;border: 2px solid gray;border-radius: 10px;border-radius: 10px;padding: 0 8px;background: LemonChiffon;selection-background-color: darkgray;padding: 6px")
        self.line5 = QLineEdit(self)
        self.line5.setGeometry(400,450,80,40)
        self.line5.setPlaceholderText("mean_smoothness ...")
        self.line5.setStyleSheet("font: bold 14px;min-width:10em;border: 2px solid gray;border-radius: 10px;border-radius: 10px;padding: 0 8px;background: LemonChiffon;selection-background-color: darkgray;padding: 6px")
        
        self.setWindowTitle('Mon Application De Classification ')
        self.show()
    def ClearFunction(self):
        self.affichage1.clear()
        self.affichage2.clear()
        self.affichage3.clear()
        self.line1.clear()
        self.line2.clear()
        self.line3.clear()
        self.line4.clear()
        self.line5.clear()
    def saveStatistique(self,tit,i,j):
            X=[i,j]
            fig = plt.figure(figsize=(7,6))
            plt.title(tit)
            dataName = ['Malignant','benign']
            explode = (0.1, 0)
            plt.pie(X, labels=dataName,explode=explode,autopct='%1.1f%%',shadow=True, startangle=90)
            fig.savefig(str(tit)+".png")
    def showDialog(self):
        home_dir = str(Path.home())
        fname = QFileDialog.getOpenFileName(self, 'Open file', home_dir)
        return fname[0]
    def traitement(self):
        data = pd.read_csv("Breast_cancer_data.csv")
        df = pd.DataFrame(data)
        x=df.iloc[:,0:5].values
        y=df.iloc[:,5].values
        x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25,random_state=0)
        i=0
        j=0
        for x in y: 
            if x==0:
                i=i+1
            else:
                j=j+1
        titre="Destribution de Données de Breast Cancer"
        self.saveStatistique(titre,i,j)
        return x_train, x_test, y_train, y_test 
    def geMainEntrer(self):
        mean_radius=float(self.line1.text())
        mean_texture=float(self.line2.text())
        mean_perimeter=float(self.line3.text())
        mean_area=float(self.line4.text())
        mean_smoothness=float(self.line5.text())
        DICTmain ={"mean_radius":mean_radius,"mean_texture":mean_texture,
                        "mean_perimeter":mean_perimeter,"mean_area":mean_area,"mean_smoothness":mean_smoothness}
        return DICTmain
        
        
    def DTreeFunction(self):
        #name = self.showDialog()
        le=self.geMainEntrer()
        print(le)
        df_test = pd.DataFrame(le,index=["DATA"])
        print(df_test)
        x_train, x_test, y_train, y_test=self.traitement()
        classifierDT =  DecisionTreeClassifier()
        classifierDT.fit(x_train,y_train)
        self.affichage1.setText("Decision Tree Classifier \n\n"+str(classification_report(y_test,classifierDT.predict(x_test))))
        if classifierDT.predict(df_test)==1:
            self.affichage2.setText("cette persone a un concer de type : Benign")
        else:
            self.affichage2.setText("cette persone a un concer de type : Malignant")
        tit="Destribution de Données de Breast Cancer"
        self.affichage3.setPixmap(QPixmap(str(tit)+".png"))
    def knnFunction(self):
        le=self.geMainEntrer()
        print(le)
        df_test = pd.DataFrame(le,index=["DATA"])
        print(df_test)
        x_train, x_test, y_train, y_test=self.traitement()
        classifierKNN =KNeighborsClassifier(n_neighbors=5)
        classifierKNN.fit(x_train,y_train)
        self.affichage1.setText("KNeighborsClassifier\n\n"+str(classification_report(y_test,classifierKNN.predict(x_test))))
        if classifierKNN.predict(df_test)==1:
            self.affichage2.setText("cette persone a un concer de type : Benign")
        else:
            self.affichage2.setText("cette persone a un concer de type : Malignant")
        tit="Destribution de Données de Breast Cancer"
        self.affichage3.setPixmap(QPixmap(str(tit)+".png"))
        
    def SVMFunction(self):
        le=self.geMainEntrer()
        print(le)
        df_test = pd.DataFrame(le,index=["DATA"])
        print(df_test)
        x_train, x_test, y_train, y_test=self.traitement()
        classifierSVM = svm.SVC(kernel='linear')
        classifierSVM.fit(x_train,y_train)
        self.affichage1.setText("Support Vector Machine Classifier\n\n"+str(classification_report(y_test,classifierSVM.predict(x_test))))
        if classifierSVM.predict(df_test)==1:
            self.affichage2.setText("cette persone a un concer de type : Benign")
        else:
            self.affichage2.setText("cette persone a un concer de type : Malignant")
        tit="Destribution de Données de Breast Cancer"
        self.affichage3.setPixmap(QPixmap(str(tit)+".png"))
        

def main():
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()