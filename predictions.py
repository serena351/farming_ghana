# import packages:

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# import data:
gh_emissions = pd.read_csv("C:/Users/seren/OneDrive/documents2/work/farming in ghana project/gh_emissions.csv")    

# split into methane and nitrous oxide datasets:
methane = gh_emissions[["year", "methane"]]
nitrous_oxide = gh_emissions[["year", "nitrous oxide"]]
    
# fit linear model:  
x = gh_emissions[["year"]]
y1 = gh_emissions[["methane"]]
y2 = gh_emissions[["nitrous oxide"]]

x = x["year"]
x = np.array(x)

y1 = y1["methane"]
y1 = np.array(y1)

y2 = y2["nitrous oxide"]
y2 = np.array(y2)

slope1, intercept1, r1, p1, std_err1 = stats.linregress(x, y1)
slope2, intercept2, r2, p2, std_err2 = stats.linregress(x, y2)

def linear(x, slope=slope1, intercept=intercept1):
  return slope * x + intercept

mymodel = list(map(linear, x))

# methane:
plt.scatter(x, y1)
plt.plot(x, mymodel)
plt.show()

# predictions:
    
newx = np.array([2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])
methane_model = list(map(linear, newx))
nitrous_model = linear(newx, slope2, intercept2)

plt.title("Prediction of greenhouse gas emissions", loc="left")
plt.xlabel("year")
plt.plot(newx, methane_model, color = "green")
plt.plot(newx, nitrous_model, color = "blue")
plt.show()

# GDP
    
# import packages:
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

# import data:
    
gdp = pd.read_csv("C:/Users/seren/OneDrive/documents2/work/farming in ghana project/gdp.csv")    
x1 = np.array(gdp["Year"])
y1 = np.array(gdp["GDP(%)"])

# check if gdp linear:

slope_gdp, int_gdp, r1, p1, std_err1 = stats.linregress(x1, y1)

def linear(x, slope=slope_gdp, intercept=int_gdp):
  return slope * x + intercept

mymodel = list(map(linear, x1))

plt.xlabel("Year")
plt.ylabel("GDP(%)")
plt.scatter(x1, y1)
plt.plot(x1, mymodel)
plt.show()

# does not look linear, use np.polyfit to fit cubic regression model:

#fit cubic regression model
cubicmodel = np.poly1d(np.polyfit(x1, y1, 3))

print(r2_score(y1, cubicmodel(x1)))
# 0.5452927453887986

#add fitted cubic regression line to scatterplot
polyline = np.linspace(2012, 2025, 15)
plt.scatter(x1, y1)
plt.plot(polyline, cubicmodel(polyline))

#add axis labels
plt.xlabel('Year')
plt.ylabel('GDP(%)')

#display plot
plt.show()

print(cubicmodel)    

newx = np.array([2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032])
ypred = cubicmodel(newx)

plt.xlabel("Year")
plt.ylabel("GDP(%)")
plt.scatter(newx, ypred)

# carbon intensity data:
    
data = {
        "year": list(gdp["Year"][:7]),
        "methane": list(gh_emissions["methane"][24:]),
        "nitrous oxide": list(gh_emissions["nitrous oxide"][24:]),
        "gdp": list(gdp["GDP(%)"][:7])
        }    
ci = pd.DataFrame(data)    

from pathlib import Path
filepath = Path("C:/Users/seren/OneDrive/documents2/work/farming in ghana project/ci.csv")  
filepath.parent.mkdir(parents=True, exist_ok=True)  
ci.to_csv(filepath)
