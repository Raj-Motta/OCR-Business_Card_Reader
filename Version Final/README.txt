Software Requirements:
	- numpy
	- pandas
	- scipy
	- matplotlib
	- opencv-python
	- jupyter
	- imutils
	- spacy
	- pytesseract

How to Install these softwares:
1. For numpy, pandas, scipy, matplotlib, opencv, jupyter and imutils:
	First, open the command line for your current file directory
	into the command line type "python3 -m venv <path for virtual environment>"
	A new folder will automatically be created that will hold all the information for your virtual environnment

	After that, type ".\<virtual environment path>\Scripts\activate" to activate your virtual environment
	(Do this every time you are going to code)
	
	When your new virtual enviornment is activated, create a .txt file in the current directory and 
	type in the names of the softwares mentioned above, save it with the name "requirements" (you can name it anything to be fair)
	
	Then, go back to your command line, and type in "pip install -r requirements.txt" and let the command line install all of the files

	And the listed softwares have been installed and are ready to use!

2. For PyTesseract:
	Go to website "https://tesseract-ocr.github.io/tessdoc/" which holds the tesseract software
	go to their github page and look for (or directly look up) "https://digi.bib.uni-mannheim.de/tesseract/"
	here you will find a list of OCRs to choose from, download the latest STABLE version of the OCR you can find
	
	after you're done downloading the tesseract installer, open the application and go through the installation process
	(Make sure to select, "Only for me", when it asks you whether you want to install it for all users or not)
	While you're selecting the folder to install your tesseract OCR, make sure to keep hold of the directory you're putting it in
	Finish the installation process
	
	After finishing the installation process, we need to check whether the OCR has been successfully installed or not, for that
	Go to your settings and search for "Edit Environment Variables for your account", and a window named "Environment Variables" will open
	On the window, look for path, and click on the edit button to check what's stored inside path, 
	you should see 2 directories with the tesseract name in it, if you do not, don't worry
	Copy the file directory you had previously saved, and add the directory here TWICE, 
	when adding it the second time add "/tessdata" to it
	That's it, your installation is complete!

3. For Spacy:
	To install spacy, look up the website "https://spacy.io/usage"
	here you'll see a lot of options to select from, and select the options that work with your system
	You'll be selecting the OS, your Platform, Package Manager (if you used pip for part 1, select pip), What hardware you'll be using
 	Configuration (Train Models), Trained Pipelines (English, and any other languages of your choice)
 	and the type of pipeline (Your choice)

	
