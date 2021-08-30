
# Program to extract number
# of rows using Python
import xlrd as xlsx

# Give the location of the file
loc = (".\\samples\\CPXPAD1.xls")
 
wb = xlsx.open_workbook(loc)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)
# Extracting number of rows
print(sheet.nrows)
for i in range(0, sheet.nrows): 
    print(sheet.cell_value(i, 0))
    #print(i)

print(sheet)
print(sheet.ncols)