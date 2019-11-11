import SimpleITK as sitk
list_dir = dir(sitk)
with open("README.md", "w") as fo:
    fo.write("# SimpleITK documentation\n\n")
    fo.write("This file was auto-generated using a Python script\n")
    fo.write("\n\n")
    for items in list_dir:
        if not (items.endswith("_swigregister") or items.startswith("_")):
            fo.write("### "+"sitk."+items)
            doc_string = "sitk."+items+".__doc__"
            print(doc_string)
            try:
                s = eval(doc_string)
                if s:
                    fo.write(s)
            except AttributeError:
                pass
                
            fo.write("\n")
			