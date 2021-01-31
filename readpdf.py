# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 14:41:03 2020

script to import tables from article PDFs into csvs

@author: dyvas
"""
import tabula
import pandas as pd
from pdf2image import convert_from_path

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

#Necessary to make tesseract OCR run, requires IDE to be in admin mode

def article(pdf,page,filename,**kwargs):
    #function to extract table from PDF article with tabula
    path = "C:/Users/dyvas/Box Sync/JournalArticles/" + pdf + ".pdf"

    dfs = tabula.read_pdf(path, pages=page,**kwargs)
    
    df = dfs[0] #convert to single dataframe
    
    outdir = './' + filename + '.csv'
    
    df.to_csv(outdir,index=False)
    
    return(df)

def articleocr(pdf,page):
    #function to extract table from PDF article with no prior OCR
    #Converts page to image, then writes OCR with tesseract, then 
    #extracts table with tabula. Likely requires additional cleanup
    path = "C:/Users/dyvas/Box Sync/JournalArticles/" + pdf + ".pdf"
    image = convert_from_path(path,first_page=page,last_page=page)
    #image[0].save(pdf + '.jpg','JPEG')

    # Get a searchable PDF
    file = pytesseract.image_to_pdf_or_hocr(image[0], extension='pdf')
    newpath = pdf + '_' + str(page) + '_ocr.pdf'
    with open(newpath, 'w+b') as f:
        f.write(file) # pdf type is bytes by default
        
    #Extract table with tabula

    dfs = tabula.read_pdf('./'+ newpath,pages=1)
    
    df = dfs[0] #convert to single dataframe
    
    outdir = './' + pdf + '_' + str(page) + '.csv'
    
    df.to_csv(outdir,index=False)
    
    return(df)

