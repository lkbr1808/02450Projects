import numpy as np
import xlrd
from datetime import datetime

# Load xlsx sheet with data
doc = xlrd.open_workbook('data.xlsx').sheet_by_index(0)

# Save attribute names
attributeNames = doc.row_values(0, 0, 11)

# Save class (month)
classLabels = doc.col_values(10, 1, 331)
classNames = sorted(set(classLabels), key=lambda date: datetime.strptime(date, '%b'))
classDict = dict(zip(classNames, range(1, 13)))

# Extract vector y
y = np.asarray([classDict[value] for value in classLabels])

# Load data into numpy array
X = np.empty((330, 9))
for i, col_id in enumerate(range(9)):
    X[:, i] = np.asarray(doc.col_values(col_id, 1, 331))

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)
