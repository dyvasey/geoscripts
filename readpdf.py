"""
Module to import tables from PDF journal articles.
"""
import tabula
import pandas as pd
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

def article(pdf,page,filename,**kwargs):
    """
    Extract table from PDF article with tabula.
    
    Requires that text be embedded in PDF. Also requires path to article
    PDFs be defined within function
    
    Parameters:
        pdf: Name of PDF file in folder defined within function
        page: Page of PDF with table
        filename: Name of .csv table to output
    
    Returns:
        df: Pandas dataframe with table
    """
    # Must set to path with PDFs on local machine
    path = "C:/Users/dyvas/Box Sync/JournalArticles/" + pdf + ".pdf"

    dfs = tabula.read_pdf(path, pages=page,**kwargs)
    
    df = dfs[0] # Convert to single dataframe
    
    outdir = './' + filename + '.csv'
    
    df.to_csv(outdir,index=False)
    
    return(df)

def articleocr(pdf,page):
    #function to extract table from PDF article with no prior OCR
    #Converts page to image, then writes OCR with tesseract, then 
    #extracts table with tabula. Likely requires additional cleanup
    
    # Add Tesseract executable to path, needed for OCR
    # Requires IDE in admin mode
    tpath = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = tpath
    
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

