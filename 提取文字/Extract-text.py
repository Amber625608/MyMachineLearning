import xml.etree.ElementTree as ET
import os
import sys
import re
from tkinter import *
from tkinter.filedialog import askdirectory
from tkinter import filedialog
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

textYW=""
textYW1=""


def trXml(element,doc,name):
    global textYW
   
    if len(element)>0:
        for child in element:
            if child.tag=="原文":
                if textYW!="":
                    print("\n",file=doc)
                textYW=child.text
                print (textYW,file=doc)
                print("",file=doc)
                state=0
                
            if child.tag=="给付条件" or child.tag=="给付金额" or child.tag=="保障范围":
                
                sp=textYW.split(child.text,1)
                if len(sp)==1:
                    dex=int(matchText(textYW,child.text))
                    print(str(dex)+","+str(dex+len(child.text))+","+child.tag,"  ",file=doc,end='')
                else:
                   
                    print(str(len(sp[0])+1)+","+str(len(sp[0])+1+len(child.text))+","+child.tag,"  ",file=doc,end='')
                
                    #防止相同字符串冲突
                    l=list(textYW)
                    l[len(sp[0]):len(sp[0])+2]='xx'
                    textYW=''.join(l)
                    l=list(textYW1)
            trXml(child,doc,name)

def matchText(s1,s2):
    n1=len(s1)
    n2=len(s2)
    for i in range(n1-n2+1):
        if fuzz.partial_ratio(s1[i:i+n2],s2)>90:#相似度90
            return int(i+1)
        
    for i in range(n1-n2+1):
        if fuzz.partial_ratio(s1[i:i+n2],s2)>50:#相似度50
            return int(i+1)

    for i in range(n1-n2+1):
        if fuzz.partial_ratio(s1[i:i+n2],s2)>20:#相似度20
            return int(i+1)

def selectInPath():
    path_ = askdirectory()
    pathIn.set(path_)
    

def selectOutPath():
    path_ = askdirectory()
    pathOut.set(path_)

def out1():
    global textYW
    for filename in os.listdir(pathIn.get()):
        print(filename)
        out(filename)
    

def out(name):
    global textYW
    print(pathIn.get()+"/"+name)
    text=open(pathIn.get()+"/"+name).read()
    text=re.sub(u"[\x00-\x08\x0b-\x0c\x0e-\x1f]+",u"",text)
    root1=ET.fromstring(text)
    PathOut=pathOut.get()+"/转换后-"+name.split('.')[0]+".txt"
    doc=open(PathOut,"w")
    print(PathOut)
    trXml(root1,doc,name)
    doc.close()
    print("done")
    textYW=""

if __name__ == "__main__":
    root = Tk()
    pathIn = StringVar()
    pathOut = StringVar()
    Label(root,text = "XML文件路径:").grid(row = 0, column = 0)
    Entry(root, textvariable  = pathIn).grid(row = 0, column = 1)
    Button(root, text = "路径选择", command = selectInPath).grid(row = 0, column = 2)
    Label(root,text = "输出路径:").grid(row = 1, column = 0)
    Entry(root, textvariable  = pathOut).grid(row = 1, column = 1)
    Button(root, text = "路径选择", command = selectOutPath).grid(row = 1, column = 2)
    Button(root, text = "确认", command = out1).grid(row = 2, column = 2)
    root.mainloop()
    

    
