# ong_pdf_reader
Example scripts for parsing coal shipping pdfs using pyocr and tesseract. Includes a 
simple cherrypy server to do fast tests and makes some image improvements using pillow 
and opencv 
## General description
From a BL pdf it parses the following topics:
* Vessel
* Port of loading
* Port of discharge
* Gross weight
* Place and date of issue (actually just the date of issue)

Bill of lading documents ussually comprises 5 pages: 3 original copies, a non negotiable 
copy and a final page with terms of service (this last page is not used for parsing)

## Troubles found (and hopefully solved!)
### Scanned documents have poor quality to injest to tesseract
When converting pdfs to images, resolution is increased to 310 dpi 
(not just 300 dpi as tesseract recomends) and all the available pages are parsed 
in order to compare the ocr results of all pages and  return only the most voted
### Captain signature and stamp might cover the place and date of issue
This is the most challenging issue. Unsharp mask filter might help in some cases
but due to the poor scanning resolution and the different scanning techniques, 
the solution is a bit complex: 
* First total pdf is OCR'ed
* If it has just one page in black and white...nothing else can be done
* If it has several pages and OCR gives same result to all, then it is clear text and nothing else needs to be done
* Otherwise the cell of "Place and date of issue" is cropped from all the images,
converted to BW, 
then aligned (split into three equal horizontal parts, to prevent the misaligment 
when some scanners deform horizontally the image) and finally a min threshold is 
applied to the image, yielding a clear image to pass to tesseract.

This process is the only robust one found to the test cases

## References
Uses code from https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python

##### Note:
A config file named config_endesa.py is needed to run tests and each file, but due to
confidentiality issues is not uploaded. The file just includes 
a dictionary with the the name of some BL files as keys and as values the parsed 
contents of each file
