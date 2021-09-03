#---
using Pkg

# Install packages
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("AutoMLPipeline")
Pkg.add("MLJ")
Pkg.add("Plots")
Pkg.add("Snappy")
Pkg.add("Queryverse")

Pkg.build("Snappy")

# Load the installed packages
using DataFrames
using CSV
using MLJ
using Plots
using Queryverse

# The pipelines
using AutoMLPipeline
using AutoMLPipeline.FeatureSelectors
using AutoMLPipeline.EnsembleMethods
using AutoMLPipeline.CrossValidators
using AutoMLPipeline.Pipelines
using AutoMLPipeline.SKPreprocessors
ENV["COLUMNS"] = 1000
ENV["ROWS"] = 10

#---
district1000 = DataFrame(CSV.File("C:/Users/arqui/Desktop/Julia_ML_Projects/engagement_data/1000.csv"))
districts = DataFrame(CSV.File("C:/Users/arqui/Desktop/Julia_ML_Projects/districts_info.csv"))
products = DataFrame(CSV.File("C:/Users/arqui/Desktop/Julia_ML_Projects/products_info.csv"))

rename!(products, "LP ID" => "lp_id");

rename!(districts, "pct_black/hispanic" => "pct_black_hispanic",
                   "pct_free/reduced" => "pct_free_reduce");

rename!(products, "Product Name" => "product_name",
                  "Provider/Company Name" => "provider_name",
                  "Sector(s)" => "sectors",
                  "Primary Essential Function" => "primary_function",
                  "Secondary Essential Function" => "secondary_function",
                  "Tertiary Essential Function" => "tertiary_function");

#---
#Clean districts_info
collect(any(ismissing.(c)) for c in eachcol(districts))
filtered = districts |> @filter(_.pct_free_reduce == NA) |> DataFrame
districtsNoNAs = districts |> @replacena(:pct_free_reduce => NaN) |> DataFrame

collect(any(ismissing.(c)) for c in eachcol(districtsNoNAs))
filtered = districtsNoNAs |> @filter(_.county_connections_ratio == NA) |> DataFrame
districtsNoNAs = districtsNoNAs |> @replacena(:county_connections_ratio => NaN) |> DataFrame

collect(any(ismissing.(c)) for c in eachcol(districtsNoNAs))
filtered = districtsNoNAs |> @filter(_.pp_total_raw == NA) |> DataFrame
districtsNoNAs = districtsNoNAs |> @replacena(:pp_total_raw => NaN) |> DataFrame

collect(any(ismissing.(c)) for c in eachcol(districtsNoNAs))

#---
#Clean products_info
collect(any(ismissing.(c)) for c in eachcol(products))
filtered = products |> @filter(_.provider_name == NA) |> DataFrame
productsNoNAs = products |> @replacena(:provider_name => NaN) |> DataFrame

collect(any(ismissing.(c)) for c in eachcol(productsNoNAs))
filtered = productsNoNAs |> @filter(_.sectors == NA) |> DataFrame
productsNoNAs = productsNoNAs |> @replacena(:sectors => NaN) |> DataFrame

collect(any(ismissing.(c)) for c in eachcol(productsNoNAs))
filtered = productsNoNAs |> @filter(_.primary_function == NA) |> DataFrame
productsNoNAs = productsNoNAs |> @replacena(:primary_function => NaN) |> DataFrame

collect(any(ismissing.(c)) for c in eachcol(productsNoNAs))
filtered = productsNoNAs |> @filter(_.secondary_function == NA) |> DataFrame
productsNoNAs = productsNoNAs |> @replacena(:secondary_function => NaN) |> DataFrame

collect(any(ismissing.(c)) for c in eachcol(productsNoNAs))
filtered = productsNoNAs |> @filter(_.tertiary_function == NA) |> DataFrame
productsNoNAs = productsNoNAs |> @replacena(:tertiary_function => NaN) |> DataFrame

collect(any(ismissing.(c)) for c in eachcol(productsNoNAs))

#---
#Opening the districts' files and combining them with their corresponding product info, and adding district id as a column
directory = walkdir("C:/Users/arqui/Desktop/Julia_ML_Projects/engagement_data/")
matchFilePath = r"[0-9][0-9][0-9][0-9]\.csv"

#Join the districts with their products
for (root, dirs, files) in directory
    for file in files
        if !(match(matchFilePath, file) == nothing)
            currentDistrict = CSV.File("C:/Users/arqui/Desktop/Julia_ML_Projects/engagement_data/" * file) |> DataFrame
            id = split(file, ".")[1]

            insertcols!(currentDistrict, 1, :district_id => id, makeunique = true)

            currentDistrictNoNAs = currentDistrict |> @replacena(:engagement_index => 0.0) |> DataFrame
            currentDistrictNoNAs = currentDistrict |> @replacena(:lp_id => 0.0) |> DataFrame
            dropmissing!(currentDistrictNoNAs)

            currentDistrict_products = innerjoin(currentDistrictNoNAs, productsNoNAs, on=["lp_id"])

            CSV.write("C:/Users/arqui/Desktop/Julia_ML_Projects/joined_engagement_data/" * "joined_" * file, currentDistrict_products)
        end
    end
end
#---
directory = walkdir("C:/Users/arqui/Desktop/Julia_ML_Projects/joined_engagement_data/")

for (root, dirs, files) in directory
    for file in files
        currentJoinedDistrict = CSV.File("C:/Users/arqui/Desktop/Julia_ML_Projects/joined_engagement_data/" * file) |> DataFrame

        currentDistrictFull = innerjoin(currentJoinedDistrict, districtsNoNAs, on = ["district_id"])

        CSV.write("C:/Users/arqui/Desktop/Julia_ML_Projects/full_engagement_data/" * "full_" * file, currentDistrictFull)
    end
end

testdf2 = innerjoin(testdf, districtsNoNAs, on = ["district_id"])
first(testdf2, 10)

#---



#describe(district1000)
#describe(districts)
#describe(projects)

#---



#---
X = data[:, [1, 2, 3, 4, 5]] |> DataFrame;

Y = data[:, 6] |> Vector;

#---
catf = CatFeatureSelector()
numf = NumFeatureSelector()

stsc = SKPreprocessor("StandardScaler")
ord = SKPreprocessor("OrdinalEncoder")
ohe = SKPreprocessor("OneHotEncoder")

forest = SKLearner("RandomForestClassifier")

#---

pipedOrd = Pipelines.@pipeline (catf |> ord)
transformedOrd = AutoMLPipeline.fit_transform!(pipedOrd, X, Y)

first(transformedOrd, 5)
#---
pipedNum = Pipelines.@pipeline (numf |> stsc)
transformedNum = AutoMLPipeline.fit_transform!(pipedNum, X, Y)

first(transformedNum, 5)
#---
modelRFC = AutoMLPipeline.@pipeline ( (numf |> stsc) + (catf |> ord) ) |> forest

fittedRFC = fit_transform!(modelRFC, X, Y)

predictions = fittedRFC == Y


scatter(Y, fittedRFC, xlabel = "True Values", ylabel = "Predicted Values")
