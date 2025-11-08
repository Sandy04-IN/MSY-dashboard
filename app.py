# user_interface.py
from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS
import pandas as pd
import re
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
from datetime import datetime, date
import numpy as np
import os
import io
import base64


app = Flask(__name__)
CORS(app)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Global storage
group = None
category = None
item = None # full uploaded dataframe (never mutated)
temp_df_preds = None        # dataframe containing actual & predicted values after building model
train_df = None
test_df = None
split_before_date = None
split_after_date = None
last_isolate_by = None
last_isolate_value = None







HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MSY Dashboard</title>
  <script src="https://unpkg.com/react@17/umd/react.development.js"></script>
  <script src="https://unpkg.com/react-dom@17/umd/react-dom.development.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <style>
    .top-bar { display:flex; justify-content:space-between; align-items:center; padding:10px; border-bottom:1px solid #ccc; }
    label { margin-right:6px; }
    select, input[type="date"] { margin-right:10px; }
  </style>
</head>
<body>
  <div id="root"></div>
  {% raw %}
  <script type="text/babel">
 
    // File Upload Tab (now auto-loads default data)
    const FileUpload = ({ setColumnNames, setTargetVariable }) => {
        const [loading, setLoading] = React.useState(false);
        const [status, setStatus] = React.useState("Loading default data...");

        React.useEffect(() => {
            const fetchDefaultData = async () => {
            setLoading(true);
            try {
                const response = await fetch("/upload", { method: "POST" }); // no file, triggers default
                const data = await response.json();

                if (data.columns) {
                setColumnNames(data.columns);
                setTargetVariable(data.columns[0]);
                setStatus(data.note || "Default data loaded successfully.");
                } else {
                setStatus(data.error || "Failed to load default data.");
                }
            } catch (error) {
                console.error("Error loading default data:", error);
                setStatus("An error occurred while loading default data.");
            } finally {
                setLoading(false);
            }
            };

            fetchDefaultData();
        }, [setColumnNames, setTargetVariable]);

        return (
            <div>
            <h2>Data Load Status</h2>
            <p>{loading ? "Loading default data..." : status}</p>
            </div>
        );
    };





    // Graphing Tab
    const Graphing = ({ modelBuilt, modelTargets }) => {
        const [x, setX] = React.useState("");
        const [y, setY] = React.useState("");
        const [groupBy, setGroupBy] = React.useState("");
        const [plotType, setPlotType] = React.useState("scatterplot");
        const [note, setNote] = React.useState("");
        const [columnOptions, setColumnOptions] = React.useState([]);
        const [imgUrl, setImgUrl] = React.useState("");
        const [mse, setMse] = React.useState(null);
        const [varVal, setVar] = React.useState(null);
        const [loadingPlot, setLoadingPlot] = React.useState(false);
        const [loadingResults, setLoadingResults] = React.useState(false);

        // Fetch columns when "Group By" changes
        React.useEffect(() => {
            if (!groupBy) {
            setColumnOptions([]);
            return;
            }

            fetch("/get_dataframe_columns", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ groupBy }),
            })
            .then((r) => r.json())
            .then((data) => {
                setColumnOptions(data.columns || []);
            })
            .catch((e) => console.error(e));
        }, [groupBy]);

        const handlePlot = async () => {
            setNote(""); // clear old warnings

            if (!x && !y) {
            alert("Please select at least one variable.");
            return;
            }

            // Validation rules
            if (plotType === "pie chart" && x && y) {
            setNote("Pie chart can only have one variable (choose either X or Y, not both).");
            return;
            }

            if (plotType !== "pie chart" && !x && !y) {
            setNote("Please select at least one variable for this plot type.");
            return;
            }

            setLoadingPlot(true);
            try {
            const response = await fetch("/plot", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ x, y, groupBy, plotType }),
            });

            if (!response.ok) {
                const txt = await response.text();
                alert("Plot error: " + txt);
                return;
            }

            const blob = await response.blob();
            setImgUrl(URL.createObjectURL(blob));
            } catch (e) {
            console.error(e);
            } finally {
            setLoadingPlot(false);
            }
        };

        // Results panel (unchanged)
        const [showResultsPanel, setShowResultsPanel] = React.useState(false);
        const [selectedDate, setSelectedDate] = React.useState("");
        const [selectedModelTarget, setSelectedModelTarget] = React.useState("");
        const [resultsImgUrl, setResultsImgUrl] = React.useState("");
        const [explainedVar, setExplainedVar] = React.useState(null);

        const handleDisplayResultsClick = () => setShowResultsPanel(true);

        const handleShowResults = async () => {
            if (!selectedDate || !selectedModelTarget) {
            alert("Please choose a date and target.");
            return;
            }

            setLoadingResults(true);
            try {
            const response = await fetch("/plot_results", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ date: selectedDate, target: selectedModelTarget }),
            });

            if (!response.ok) {
                const txt = await response.text();
                alert("Error: " + txt);
                return;
            }

            const data = await response.json();
            if (!data.image) {
                alert("No image returned from server.");
                return;
            }

            setResultsImgUrl("data:image/png;base64," + data.image);
            setMse(typeof data.mse === "number" ? data.mse.toFixed(4) : "N/A");
            setVar(typeof data.variance === "number" ? data.variance.toFixed(4) : "N/A");
            setExplainedVar(
                typeof data.explained_variance === "number"
                ? data.explained_variance.toFixed(4)
                : "N/A"
            );
            } catch (e) {
            console.error(e);
            } finally {
            setLoadingResults(false);
            }
        };

        return (
            <div>
            <h2>Graphing</h2>

            {!modelBuilt && (
                <>
                <div>
                    <label>Group By: </label>
                    <select value={groupBy} onChange={(e) => setGroupBy(e.target.value)}>
                    <option value="">--Select--</option>
                    <option value="Group">Group</option>
                    <option value="Category">Category</option>
                    <option value="Item">Item</option>
                    </select>
                </div>

                {groupBy && (
                    <>
                    <div style={{ marginTop: 10 }}>
                        <label>Plot Type: </label>
                        <select value={plotType} onChange={(e) => setPlotType(e.target.value)}>
                        <option value="barplot">Barplot</option>
                        <option value="scatterplot">Scatterplot</option>
                        <option value="line plot">Line Plot</option>
                        <option value="pie chart">Pie Chart</option>
                        </select>
                    </div>

                    <div style={{ marginTop: 10 }}>
                        <label>X: </label>
                        <select value={x} onChange={(e) => setX(e.target.value)}>
                        <option value="">--Select--</option>
                        {columnOptions.map((col) => (
                            <option key={col} value={col}>
                            {col}
                            </option>
                        ))}
                        </select>

                        <label> Y: </label>
                        <select value={y} onChange={(e) => setY(e.target.value)}>
                        <option value="">--Select--</option>
                        {columnOptions.map((col) => (
                            <option key={col} value={col}>
                            {col}
                            </option>
                        ))}
                        </select>
                    </div>
                    </>
                )}

                {note && (
                    <div style={{ marginTop: "8px", color: "darkorange", fontWeight: "bold" }}>
                    {note}
                    </div>
                )}

                <div style={{ marginTop: "10px" }}>
                    <button onClick={handlePlot} disabled={loadingPlot || !groupBy}>
                    {loadingPlot ? "Plotting..." : "Generate Plot"}
                    </button>
                </div>

                {imgUrl && (
                    <div style={{ marginTop: "20px" }}>
                    <img src={imgUrl} alt="Plot" style={{ maxWidth: "100%" }} />
                    </div>
                )}
                </>
            )}

            {modelBuilt && (
                <div style={{ marginTop: 10 }}>
                <button onClick={handleDisplayResultsClick}>Display Results</button>
                </div>
            )}

            {modelBuilt && showResultsPanel && (
                <div style={{ marginTop: 12 }}>
                <div>
                    <label>Date: </label>
                    <input
                    type="date"
                    value={selectedDate}
                    onChange={(e) => setSelectedDate(e.target.value)}
                    />
                    <label> Target: </label>
                    <select
                    value={selectedModelTarget}
                    onChange={(e) => setSelectedModelTarget(e.target.value)}
                    >
                    <option value="">--Select--</option>
                    {modelTargets.map((t, i) => (
                        <option key={i} value={t}>
                        {t}
                        </option>
                    ))}
                    </select>
                    <button
                    onClick={handleShowResults}
                    disabled={loadingResults}
                    style={{ marginLeft: 10 }}
                    >
                    {loadingResults ? "Plotting..." : "Show Results"}
                    </button>
                </div>

                {resultsImgUrl && (
                    <div
                    style={{
                        display: "flex",
                        alignItems: "flex-start",
                        marginTop: 16,
                        gap: "2rem",
                    }}
                    >
                    <img src={resultsImgUrl} alt="Results" style={{ maxWidth: "65%" }} />
                    <div style={{ fontSize: "1rem", lineHeight: "1.6" }}>
                        <strong>MSE:</strong> {mse}
                        <br />
                        <strong>Variance:</strong> {varVal}
                        <br />
                        <strong>Explained Variance:</strong> {explainedVar}
                    </div>
                    </div>
                )}
                </div>
            )}
            </div>
        );
    };




    // Application
    const App = () => {
      const [activeTab, setActiveTab] = React.useState("File Upload");
      const [columnNames, setColumnNames] = React.useState([]);
      const [targetVariable, setTargetVariable] = React.useState("");
      const [modelBuilt, setModelBuilt] = React.useState(false);
      const [modelTargets, setModelTargets] = React.useState([]);


      const handleModelBuilt = (targets) => {
        setModelBuilt(true);
        setModelTargets(targets || []);


        setActiveTab("Graphing");
      };


      return (
        <div>
          <div className="top-bar">
            <div>
              <button onClick={() => setActiveTab("File Upload")}>File Upload</button>
              <button onClick={() => setActiveTab("Graphing")}>Graphing</button>
            </div>
            <div>
              {/* could show modelBuilt indicator */}
              {modelBuilt ? <strong>Model built</strong> : null}
            </div>
          </div>


          <div style={{ padding: 12 }}>
            {activeTab === "File Upload" && (
              <FileUpload setColumnNames={setColumnNames} setTargetVariable={setTargetVariable} />
            )}
            {activeTab === "Graphing" && (
              <Graphing columnNames={columnNames} modelBuilt={modelBuilt} modelTargets={modelTargets} />
            )}
          </div>
        </div>
      );
    };


    ReactDOM.render(<App />, document.getElementById('root'));
  </script>
  {% endraw %}
</body>
</html>
"""


# Flask endpoints


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/upload", methods=["POST"])
def upload_file():
    global group, category, item

    try:
        import re

        # loading excel files
        months = [
            "data/May_Data_Matrix (1).xlsx",
            "data/June_Data_Matrix.xlsx",
            "data/July_Data_Matrix (1).xlsx",
            "data/August_Data_Matrix (1).xlsx",
            "data/September_Data_Matrix.xlsx",
            "data/October_Data_Matrix_20251103_214000.xlsx",
        ]

        dfs = []
        for fpath in months:
            for sheet_idx in range(3):
                df = pd.read_excel(fpath, sheet_name=sheet_idx)
                df["month"] = (
                    os.path.basename(fpath).split("_")[0]
                    .replace("Data", "")
                    .strip()
                )
                dfs.append(df)

        # splitting into categories
        group_dfs, category_dfs, item_dfs = [], [], []
        for df in dfs:
            cols = df.columns
            if "Group" in cols:
                group_dfs.append(df)
            elif "Category" in cols:
                category_dfs.append(df)
            else:
                item_dfs.append(df)

        group = pd.concat(group_dfs, ignore_index=True)
        category = pd.concat(category_dfs, ignore_index=True)
        item = pd.concat(item_dfs, ignore_index=True)

        group["type"] = "Group"
        category["type"] = "Category"
        item["type"] = "Specific Item"

        # numeric conversions and cost calculation
        for df in [group, category, item]:
            for col in ["Amount", "Count"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col]
                        .astype(str)
                        .str.replace(r"[^0-9.\-]", "", regex=True),
                        errors="coerce",
                    )
            df["cost"] = (df.get("Amount", 0) / df.get("Count", 1)).fillna(0)

        # loading ingredient and shipment data
        ing = pd.read_csv("data/MSY Data - Ingredient.csv")
        ship = pd.read_csv("data/MSY Data - Shipment.csv")

        # merging items and ingredients
        item["Item Name"] = item["Item Name"].astype(str)
        ing["Item name"] = ing["Item name"].astype(str)

        item_ing = pd.merge(
            item,
            ing,
            left_on=item["Item Name"].str.strip().str.lower(),
            right_on=ing["Item name"].str.strip().str.lower(),
            how="outer",
            suffixes=("_item", "_ing"),
        )

        item_ing.rename(
            columns={"Item name": "Items with Ingredient Counts"}, inplace=True
        )
        item_ing.columns = item_ing.columns.str.strip()

        # helper function to normalize text
        def normalize_text(s):
            if isinstance(s, str):
                s = s.strip().lower()
                s = re.sub(r"[^a-z]+$", "", s)
                return s
            return s

        normalized_columns = [normalize_text(col) for col in item_ing.columns]

        # merging shipment data
        ship_copy = ship.copy()
        ship_copy["Ingredient"] = ship_copy["Ingredient"].astype(str)

        def expand_ingredient(ing_name):
            parts = re.split(r"\s*\+\s*", ing_name)
            return [normalize_text(p) for p in parts if p]

        merged = item_ing.copy()

        for _, row in ship_copy.iterrows():
            ingredients = expand_ingredient(row["Ingredient"])
            matched = False

            for ingredient in ingredients:
                matching_cols = [
                    col for col in normalized_columns if ingredient in col and ingredient != ""
                ]
                if matching_cols:
                    ship_rows = ship_copy[
                        ship_copy["Ingredient"].str.lower().str.contains(ingredient, na=False)
                    ]
                    merged = pd.concat([merged, ship_rows], axis=0, ignore_index=True)
                    matched = True

            if not matched:
                print(f"No matching column found for ingredient(s): {ingredients}")

        merged.reset_index(drop=True, inplace=True)
        item_ing = merged

        # filling nas
        group.fillna({"cost": 0}, inplace=True)
        category.fillna({"cost": 0}, inplace=True)
        item_ing.fillna({"cost": 0}, inplace=True)
        
        item = item_ing.copy()


        # return success
        return jsonify({
            "columns": list(item.columns),
            "note": "Default data loaded and merged successfully."
        })

    except Exception as e:
        return jsonify({"error": f"Failed to load default data: {str(e)}"}), 500



 
@app.route("/get_dataframe_columns", methods=["POST"])
def get_dataframe_columns():
    global group, category, item
    req = request.json or {}
    groupBy = req.get("groupBy")

    df_map = {
        "Group": group,
        "Category": category,
        "Item": item
    }

    df = df_map.get(groupBy)
    if df is None:
        return jsonify({"columns": []})

    columns = list(df.columns)
    return jsonify({"columns": columns})



@app.route("/plot", methods=["POST"])
def plot():
    global group, category, item
    req = request.json or {}
    x = req.get("x")
    y = req.get("y")
    groupBy = req.get("groupBy")
    plot_type = req.get("plotType", "scatterplot").lower()

    df_map = {
        "Group": group,
        "Category": category,
        "Item": item
    }

    df_temp = df_map.get(groupBy)
    if df_temp is None:
        return "Invalid groupBy selection", 400

    if plot_type == "pie chart" and x and y:
        return "Pie chart can only have one variable (choose either X or Y).", 400

    plt.figure(figsize=(8, 6))
    try:
        if plot_type == "scatterplot":
            if x and y:
                sns.scatterplot(data=df_temp, x=x, y=y).tick_params(axis='x', labelrotation=90)
                plt.title(f"Scatterplot: {y} vs {x} ({groupBy})")
            else:
                return "Scatterplot requires both X and Y variables.", 400

        elif plot_type == "barplot":
            if x and y:
                sns.barplot(data=df_temp, x=x, y=y).tick_params(axis='x', labelrotation=90)
                plt.title(f"Barplot of {y} by {x} ({groupBy})")
            else:
                return "Barplot requires both X and Y variables.", 400

        elif plot_type == "line plot":
            if x and y:
                sns.lineplot(data=df_temp, x=x, y=y).tick_params(axis='x', labelrotation=90)
                plt.title(f"Line Plot of {y} vs {x} ({groupBy})")
            else:
                return "Line plot requires both X and Y variables.", 400

        elif plot_type == "pie chart":
            var = x or y
            if not var:
                return "Pie chart requires one variable.", 400
            counts = df_temp[var].value_counts()
            plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%')
            plt.title(f"Pie Chart of {var} ({groupBy})")

        else:
            return f"Invalid plot type '{plot_type}'", 400

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close('all')
        return send_file(buf, mimetype="image/png")

    except Exception as e:
        plt.close('all')
        return str(e), 500





if __name__ == "__main__":
    app.run(debug=True, port = 5001)