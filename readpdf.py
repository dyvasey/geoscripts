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
    """
    Extract table from PDF with no embedded text using OCR and tabula.
    
    Function uses Tesseract to implement optical character recognition (OCR),
    then extracts table with tabula. Requires Tesseract to be installed 
    locally with executable path defined in function. Resulting tables likely
    to have errors and need additional cleanup.
    
    Parameters:
        pdf: Name of PDF file in folder defined within function
        page: Page of PDF with table
    
    Returns:
        df: Pandas dataframe with table
    """
    # Add Tesseract executable to path, needed for OCR.
    # Requires IDE in admin mode
    tpath = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = tpath
    
    # Must set to path with PDFs on local machine
    path = "C:/Users/dyvas/Box Sync/JournalArticles/" + pdf + ".pdf"
    
    # Convert PDF page to image
    image = convert_from_path(path,first_page=page,last_page=page)

    # Get a searchable PDF
    file = pytesseract.image_to_pdf_or_hocr(image[0], extension='pdf')
    newpath = pdf + '_' + str(page) + '_ocr.pdf'
    with open(newpath, 'w+b') as f:
        f.write(file) # PDF type is bytes by default
        
    #Extract table with tabula
    dfs = tabula.read_pdf('./'+ newpath,pages=1)
    df = dfs[0] # Convert to single dataframe
    
    # Save .csv with extracted table
    outdir = './' + pdf + '_' + str(page) + '.csv'
    df.to_csv(outdir,index=False)
    
    return(df)

def imgocr(img):
    """
    Extract table from JPG image using OCR and tabula.
    
    Function uses Tesseract to implement optical character recognition (OCR),
    then extracts table with tabula. Requires Tesseract to be installed 
    locally with executable path defined in function. Resulting tables likely
    to have errors and need additional cleanup.
    
    Parameters:
        img: Name of image file in current directory
    
    Returns:
        df: Pandas dataframe with table
    """
    # Add Tesseract executable to path, needed for OCR.
    # Requires IDE in admin mode
    tpath = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = tpath
    
    # Open the image
    image = Image.open(img)

    # Get a searchable PDF
    file = pytesseract.image_to_pdf_or_hocr(image, extension='pdf')
    newpath = img[0:-4] + '_ocr.pdf'
    with open(newpath, 'w+b') as f:
        f.write(file) # PDF type is bytes by default
        
    #Extract table with tabula
    dfs = tabula.read_pdf('./'+ newpath,pages=1)
    df = dfs[0] # Convert to single dataframe
    
    # Save .csv with extracted table
    outdir = './' + img[0:-4] + '.csv'
    df.to_csv(outdir,index=False)
    
    return(df)

