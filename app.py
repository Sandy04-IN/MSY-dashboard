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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, explained_variance_score
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
item = None
ship = None
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
            // 🔥 FIX: Clear previous selections when the grouping category changes
            setX("");
            setY("");

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
                    <option value="Ingredient/Shipment">Ingredient/Shipment</option>
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


    // Next Month Usage Tab
    const NextMonthUsage = () => {
        const [monthToPredict, setMonthToPredict] = React.useState("");
        const [predictionResults, setPredictionResults] = React.useState(null);
        const [loading, setLoading] = React.useState(false);
        const [imgUrl, setImgUrl] = React.useState("");
        const [note, setNote] = React.useState("");

        const MONTH_OPTIONS = ["June", "July", "Aug", "Sept", "Oct", "Nov"];

        const handlePredict = async () => {
            if (!monthToPredict) {
                alert("Please select a month to predict.");
                return;
            }

            setLoading(true);
            setPredictionResults(null);
            setImgUrl("");
            setNote("");

            try {
                const response = await fetch("/predict_next_month_usage", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ month: monthToPredict }),
                });

                if (!response.ok) {
                    const txt = await response.text();
                    setNote("Prediction Error: " + txt);
                    return;
                }

                const data = await response.json();
                
                if (data.error) {
                    setNote(data.error);
                    return;
                }
                
                // Handle image data separately if available
                if (data.image) {
                    setImgUrl("data:image/png;base64," + data.image);
                    delete data.image; // Clean up the JSON data for display
                }
                
                setPredictionResults(data);
                setNote(data.note || "Prediction successful!");

            } catch (e) {
                console.error("Prediction error:", e);
                setNote("An unexpected error occurred during prediction.");
            } finally {
                setLoading(false);
            }
        };

        return (
            <div>
                <h2>Next Month Usage Prediction</h2>

                <div style={{ marginBottom: 15 }}>
                    <label>Month to Predict: </label>
                    <select 
                        value={monthToPredict} 
                        onChange={(e) => setMonthToPredict(e.target.value)}
                    >
                        <option value="">--Select--</option>
                        {MONTH_OPTIONS.map((month) => (
                            <option key={month} value={month}>
                                {month}
                            </option>
                        ))}
                    </select>

                    <button 
                        onClick={handlePredict} 
                        disabled={loading || !monthToPredict}
                        style={{ marginLeft: 10 }}
                    >
                        {loading ? "Predicting..." : "Run Prediction"}
                    </button>
                </div>

                {note && (
                    <div style={{ color: predictionResults?.error ? "red" : "black", marginBottom: 15 }}>
                        <strong>{note}</strong>
                    </div>
                )}

                {predictionResults && (
                    <div style={{ marginTop: 20 }}>
                        <h3>Prediction Results</h3>
                        <p>
                            <strong>Mean Squared Error (MSE):</strong> {typeof predictionResults.mse === "number" ? predictionResults.mse.toFixed(4) : predictionResults.mse || 'N/A'}
                        </p>
                        <p>
                            <strong>Variance:</strong> {typeof predictionResults.variance === "number" ? predictionResults.variance.toFixed(4) : predictionResults.variance || 'N/A'}
                        </p>
                        <p>
                            <strong>Explained Variance:</strong> {typeof predictionResults.explained_variance === "number" ? predictionResults.explained_variance.toFixed(4) : predictionResults.explained_variance || 'N/A'}
                        </p>
                        
                        {imgUrl && (
                            <div style={{ marginTop: 20 }}>
                                <img src={imgUrl} alt="Actual vs Predicted Usage" style={{ maxWidth: "100%" }} />
                            </div>
                        )}
                    </div>
                )}
            </div>
        );
    };
    
    // Shipment vs. Usage Tab
    const ShipmentVsUsage = () => {
        const [selectedMonth, setSelectedMonth] = React.useState("");
        const [monthOptions, setMonthOptions] = React.useState([]);
        const [imgUrl, setImgUrl] = React.useState("");
        const [note, setNote] = React.useState("");
        const [loading, setLoading] = React.useState(false);

        // Fetch unique months for the dropdown on component load
        React.useEffect(() => {
            fetch("/get_unique_months")
                .then((r) => r.json())
                .then((data) => {
                    setMonthOptions(data.months || []);
                })
                .catch((e) => console.error("Error fetching months:", e));
        }, []);

        const handleGenerateChart = async () => {
            if (!selectedMonth) {
                setNote("Please select a month.");
                return;
            }

            setLoading(true);
            setImgUrl("");
            setNote("");

            try {
                const response = await fetch("/shipment_vs_usage_plot", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ month: selectedMonth }),
                });

                if (!response.ok) {
                    const txt = await response.text();
                    setNote("Chart Error: " + txt);
                    return;
                }

                const data = await response.json();

                if (data.error) {
                    setNote("Error: " + data.error);
                    return;
                }
                
                if (data.image) {
                    setImgUrl("data:image/png;base64," + data.image);
                    setNote(data.note || "Chart generated successfully.");
                }

            } catch (e) {
                console.error("Plotting error:", e);
                setNote("An unexpected error occurred during plotting.");
            } finally {
                setLoading(false);
            }
        };

        return (
            <div>
                <h2>Shipment vs. Usage Analysis</h2>

                <div style={{ marginBottom: 15 }}>
                    <label>Month: </label>
                    <select 
                        value={selectedMonth} 
                        onChange={(e) => setSelectedMonth(e.target.value)}
                    >
                        <option value="">--Select Month--</option>
                        {monthOptions.map((month) => (
                            <option key={month} value={month}>
                                {month}
                            </option>
                        ))}
                    </select>

                    <button 
                        onClick={handleGenerateChart} 
                        disabled={loading || !selectedMonth}
                        style={{ marginLeft: 10 }}
                    >
                        {loading ? "Generating..." : "Generate Chart"}
                    </button>
                </div>

                {note && (
                    <div style={{ color: "darkblue", marginBottom: 15 }}>
                        <strong>{note}</strong>
                    </div>
                )}

                {imgUrl && (
                    <div style={{ marginTop: 20 }}>
                        <img src={imgUrl} alt="Shipment vs Usage Plot" style={{ maxWidth: "100%" }} />
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
              <button onClick={() => setActiveTab("NextMonthUsage")}>Next Month Usage</button>
              <button onClick={() => setActiveTab("ShipmentVsUsage")}>Shipment vs. Usage</button>
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
            {activeTab === "NextMonthUsage" && (
              <NextMonthUsage columnNames={columnNames} modelBuilt={modelBuilt} modelTargets={modelTargets} />
            )}
            {activeTab === "ShipmentVsUsage" && (
                <ShipmentVsUsage columnNames={columnNames} modelBuilt={modelBuilt} modelTargets={modelTargets} />
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
    global group, category, item, ship

    try:
        import re
        import pandas as pd
        import os
        from flask import jsonify
        import numpy as np

        # --- Initial Data Loading ---

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

        # Split by type
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

        # Numeric conversions
        for df in [group, category, item]:
            for col in ["Amount", "Count"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
                        errors="coerce",
                    )
            df["cost"] = (df.get("Amount", 0) / df.get("Count", 1)).fillna(0)

        # Load external data
        ing = pd.read_csv("data/MSY Data - Ingredient.csv")
        ship = pd.read_csv("data/MSY Data - Shipment.csv")

        # --- Merge item and ingredient datasets ---
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

        # --- Fix ship total calculation ---
        ship["total"] = (
            ship["Quantity per shipment"].astype(int)
            * ship["Number of shipments"].astype(int)
        )

        # Iterate properly
        #for row in ship.itertuples():
        #    freq = str(row.frequency).strip().lower()
        #    if freq == "weekly":
        #        ship.at[row.Index, "total"] *= 4  # Approximate 4 weeks in a month
        #    elif freq == "biweekly":
        #        ship.at[row.Index, "total"] *= 2  # Approximate 2 biweeks in a month

        # --- Drop NaNs from categorical columns ---
        def drop_categorical_nans(df):
            non_numeric_cols = df.select_dtypes(
                include=["object", "category", "bool", "datetime"]
            ).columns
            df_cleaned = df.dropna(subset=non_numeric_cols)
            print(f"Dropped {len(df) - len(df_cleaned)} rows due to categorical NaNs.")
            return df_cleaned

        group = drop_categorical_nans(group)
        category = drop_categorical_nans(category)
        item_ing = drop_categorical_nans(item_ing)

        # --- Fill NaNs ---
        group.fillna(0, inplace=True)
        category.fillna(0, inplace=True)
        item_ing.fillna(0, inplace=True)

        # --- Map month numbers ---
        month_map = {
            "May": 5,
            "June": 6,
            "July": 7,
            "August": 8,
            "September": 9,
            "October": 10,
        }

        for df in [group, category, item_ing]:
            df["month numerical"] = df["month"].map(month_map).fillna(0).astype(int)

        # Assign final item data
        item = item_ing.copy()

        # --- Save processed files ---
        group.to_csv("data/processed/group.csv", index=False)
        category.to_csv("data/processed/category.csv", index=False)
        item.to_csv("data/processed/item.csv", index=False)
        ship.to_csv("data/processed/ship.csv", index=False)

        # Return success
        return jsonify(
            {
                "columns": list(item.columns),
                "note": "Default data loaded and merged successfully.",
            }
        )

    except Exception as e:
        return jsonify({"error": f"Failed to load default data: {str(e)}"}), 500



 
@app.route("/get_dataframe_columns", methods=["POST"])
def get_dataframe_columns():
    global group, category, item, ing_ship
    req = request.json or {}
    groupBy = req.get("groupBy")

    df_map = {
        "Group": group,
        "Category": category,
        "Item": item,
        "Ingredient/Shipment": ing_ship
    }

    df = df_map.get(groupBy)
    if df is None:
        return jsonify({"columns": []})

    columns = list(df.columns)
    return jsonify({"columns": columns})



@app.route("/plot", methods=["POST"])
def plot():
    global group, category, item, ing_ship
    req = request.json or {}
    x = req.get("x")
    y = req.get("y")
    groupBy = req.get("groupBy")
    plot_type = req.get("plotType", "scatterplot").lower()

    df_map = {
        "Group": group,
        "Category": category,
        "Item": item,
        "Ingredient/Shipment": ing_ship
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




@app.route("/predict_next_month_usage", methods=["POST"])
def predict_next_month_usage():
    global item # Use the global item DataFrame

    # Assuming normalize_text is available globally or imported
    def normalize_text(s):
        if isinstance(s, str):
            s = s.strip().lower()
            s = re.sub(r"[^a-z]+$", "", s)
            s = re.sub(r"[^a-z]+", "", s) 
            return s
        return s
    
    if item is None or item.empty:
        return jsonify({"error": "Item data not loaded. Please upload data first."}), 400

    req = request.json or {}
    month_to_predict_str = req.get("month")

    if not month_to_predict_str:
        return jsonify({"error": "Month to predict is required."}), 400

    # Map month names to numerical order for filtering
    month_map = {
        "may": 5, "june": 6, "july": 7, "aug": 8, "sept": 9, "oct": 10, "nov": 11
    }
    month_num = month_map.get(month_to_predict_str.lower())
    
    # Map selected month string to actual month name in the dataset (for test_df lookup)
    # Corrected months for dataset lookup based on user input (Aug -> August, Sept -> September, etc.)
    month_actual_name_map = {
        "june": "June", "july": "July", "aug": "August", "sept": "September", "oct": "October", "nov": "November"
    }
    month_for_test = month_actual_name_map.get(month_to_predict_str.lower())


    if month_num is None:
        return jsonify({"error": f"Invalid month selected: {month_to_predict_str}"}), 400

    # Define target ingredients (these are the normalized column names)
    targets = [
        "braisedbeefusedg", "braisedchickeng", "braisedporkg", "eggcount", 
        "riceg", "ramencount", "ricenoodlesg", "chickenthighpcs", 
        "chickenwingspcs", "flourg", "picklecabbage", "greenonion", 
        "cilantro", "whiteonion", "peasg", "carrotg", "bokchoyg", "tapiocastarch"
    ]
    
    # 1. TEMPORARILY NORMALIZE COLUMN NAMES
    
    # Create a mapping of old column names to new normalized column names
    col_name_map = {col: normalize_text(col) for col in item.columns}
    
    # Apply temporary rename to a DEEP COPY of the item DataFrame. 
    df_pred_base = item.copy()
    df_pred_base.columns = df_pred_base.columns.map(col_name_map)
    
    # Ensure 'month' column is handled consistently
    if 'month' not in df_pred_base.columns:
         return jsonify({"error": "Could not find 'month' column after normalization."}), 400
         
    df_pred_base['month_num'] = df_pred_base['month'].astype(str).str.lower().map(month_map)
    
    # 2. Filter the training data: Use all months BEFORE the selected month (using month_num)
    train_df = df_pred_base[df_pred_base['month_num'] < month_num].copy()
    
    # 3. Filter the test data: Use the data ONLY from the month to predict (using month_actual_name_map)
    # The filter uses the original, non-normalized 'month' column content (e.g., 'August')
    if month_for_test:
        test_df = df_pred_base[df_pred_base['month'].astype(str) == month_for_test].copy()
    else:
        # For November (month_for_test is None), initialize an empty DataFrame
        test_df = df_pred_base.head(0).copy() 


    if train_df.empty:
        return jsonify({"error": f"No historical data available before {month_to_predict_str} to build the model."}), 400

    # Ensure all target columns are numeric and fill NaNs with 0 for model input
    for col in targets:
        if col not in train_df.columns:
             return jsonify({"error": f"Required ingredient column '{col}' not found in the item dataset."}), 400
        
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        if not test_df.empty:
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)


    # Select all *other* numeric columns as predictors (excluding month_num and the targets)
    exclude_cols = set(targets + ['month_num'])
    predictors = [
        col for col in train_df.select_dtypes(include='number').columns 
        if col not in exclude_cols
    ]

    if not predictors:
        return jsonify({"error": "No other numeric columns available to use as predictors."}), 400

    
    # --- Stepwise Regression and Prediction ---
    
    predictions = {}
    actuals = {}
    total_mse = 0
    total_actual_var = 0
    valid_targets_count = 0
    
    # X_train setup (Predictors)
    X_train = train_df[predictors].fillna(train_df[predictors].mean())
    train_means = X_train.mean().to_dict()

    for target in targets:
        y_train = train_df[target]
        
        # Stepwise Selection (Both directions, using AIC)
        selected_features = []
        remaining = list(X_train.columns)
        best_score = float("inf")
        
        while True:
            # --- 1. FORWARD STEP (Adding a predictor) ---
            scores_to_add = []
            
            # Filter remaining candidates for stability
            valid_candidates_to_add = [
                c for c in remaining if X_train[c].nunique() > 1
            ]
            
            for candidate in valid_candidates_to_add:
                try:
                    features = selected_features + [candidate]
                    model_try = sm.OLS(y_train, sm.add_constant(X_train[features], has_constant="add")).fit()
                    score = model_try.aic 
                    scores_to_add.append((score, candidate))
                except (np.linalg.LinAlgError, Exception):
                    continue

            # --- 2. BACKWARD STEP (Removing a predictor) ---
            scores_to_remove = []
            
            if selected_features:
                for candidate in selected_features:
                    try:
                        features = [f for f in selected_features if f != candidate]
                        # Must check if constant can be added if features list becomes empty
                        if not features:
                            # Model with only constant
                            model_try = sm.OLS(y_train, sm.add_constant(pd.DataFrame(index=X_train.index), has_constant="add")).fit()
                        else:
                            model_try = sm.OLS(y_train, sm.add_constant(X_train[features], has_constant="add")).fit()
                            
                        score = model_try.aic 
                        scores_to_remove.append((score, candidate))
                    except (np.linalg.LinAlgError, Exception):
                        continue

            # --- 3. DECISION STEP ---
            
            # Find the best move (lowest score from adding or removing)
            best_forward = min(scores_to_add) if scores_to_add else (float("inf"), None)
            best_backward = min(scores_to_remove) if scores_to_remove else (float("inf"), None)

            current_best_score = min(best_forward[0], best_backward[0])
            
            if current_best_score < best_score:
                # A move improved the model score
                if current_best_score == best_forward[0]:
                    # Best move is to add a feature (Forward)
                    best_candidate = best_forward[1]
                    selected_features.append(best_candidate)
                    if best_candidate in remaining:
                        remaining.remove(best_candidate)
                    # print(f"Added {best_candidate}. New AIC: {current_best_score:.4f}") # Debugging
                else:
                    # Best move is to remove a feature (Backward)
                    best_candidate = best_backward[1]
                    selected_features.remove(best_candidate)
                    remaining.append(best_candidate) # Move the removed feature back to 'remaining'
                    # print(f"Removed {best_candidate}. New AIC: {current_best_score:.4f}") # Debugging
                    
                best_score = current_best_score
            else:
                # No move improved the model score, stop iteration
                break
            
            
        # --- Model Prediction ---
        
        # If test_df is empty (November), create dummy X_test with training means to get a prediction
        if test_df.empty:
            # Create a single dummy row for prediction, filled with training means
            X_test_dummy = pd.DataFrame([train_means], index=[month_to_predict_str])
            X_test_base = X_test_dummy[selected_features].copy()
            y_train_min, y_train_max = y_train.min(), y_train.max()
        else:
            X_test_base = test_df[selected_features].copy()
            y_train_min, y_train_max = y_train.min(), y_train.max()

        # Handle prediction based on selected features
        if not selected_features:
            predictions[target] = pd.Series(y_train.mean(), index=X_test_base.index if not X_test_base.empty else [month_to_predict_str])
            final_model = None
        else:
            try:
                # Final model fitting (using the training data)
                final_model = sm.OLS(y_train, sm.add_constant(X_train[selected_features], has_constant="add")).fit()
                
                # Fill missing test data with training means
                for col_name, mean_val in train_means.items():
                    if col_name in X_test_base.columns:
                        X_test_base[col_name] = X_test_base[col_name].fillna(mean_val)
                
                # Prepare X_test for prediction
                X_test = sm.add_constant(X_test_base, has_constant="add")
                X_test = X_test.reindex(columns=final_model.model.exog_names, fill_value=0.0).astype(float)
                
                preds = final_model.predict(X_test)
                
                # Clip predictions to training range
                preds = preds.clip(lower=y_train_min, upper=y_train_max)
                predictions[target] = preds

            except Exception as e:
                # Prediction failed: use the mean
                predictions[target] = pd.Series(y_train.mean(), index=X_test_base.index if not X_test_base.empty else [month_to_predict_str])
                print(f"Prediction failed for {target}: {e}")
                final_model = None

        
        # Aggregate metrics for valid test month data
        if not test_df.empty:
            actual = test_df[target]
            predicted = predictions[target]
            
            # Align actual and predicted series by index and drop NaNs
            actual_aligned, predicted_aligned = actual.align(predicted, join='inner', fill_value=np.nan)
            valid_idx = actual_aligned.dropna().index.intersection(predicted_aligned.dropna().index)
            
            actual_aligned = actual_aligned.loc[valid_idx]
            predicted_aligned = predicted_aligned.loc[valid_idx]
            
            if not actual_aligned.empty:
                total_mse += mean_squared_error(actual_aligned, predicted_aligned)
                total_actual_var += np.var(actual_aligned)
                valid_targets_count += 1
                
                actuals[target] = actual_aligned.sum()
                predictions[target] = predicted_aligned.sum()
            else:
                actuals[target] = 0
                predictions[target] = predicted.sum() # Predicted sum of the non-empty series
        else:
            # For prediction-only months (e.g., November)
            actuals[target] = np.nan 
            predictions[target] = predictions[target].sum()

    # --- Final Metric Calculation ---
    
    # November or any prediction-only month will have test_df.empty == True
    if month_to_predict_str.lower() == 'nov' or test_df.empty:
        avg_mse, avg_var, explained_var = "N/A", "N/A", "N/A"
        note = f"Prediction successful for {month_to_predict_str}."
    elif valid_targets_count > 0:
        # Calculate overall metrics
        avg_mse = total_mse / valid_targets_count
        avg_var = total_actual_var / valid_targets_count
        explained_var = 1 - (avg_mse / avg_var) if avg_var and avg_var != 0 else 0
        note = f"Stepwise prediction for {month_to_predict_str} completed successfully."
    else:
        avg_mse, avg_var, explained_var = np.nan, np.nan, np.nan
        note = f"Prediction failed to produce valid metrics for any target."


    # --- Plotting ---
    
    # Prepare data for plotting (Actual vs. Predicted Sums for ALL ingredients)
    plot_data = pd.DataFrame({
        'Actual Usage': actuals,
        'Predicted Usage': {k: v for k, v in predictions.items() if not isinstance(v, pd.Series)}
    })
    
    # Replace NaN actuals with 0 for plotting consistency
    plot_data['Actual Usage'] = plot_data['Actual Usage'].fillna(0)
    
    # Ensure import is inside function if it wasn't global
    import matplotlib.pyplot as plt 
    import io, base64 
    
    plt.figure(figsize=(12, 6))
    plot_data.plot(kind='bar', ax=plt.gca(), width=0.8)
    
    plt.title(f"Actual vs. Predicted Ingredient Usage (Total for {month_to_predict_str})")
    plt.ylabel("Total Usage (Sum)")
    plt.xlabel("Ingredient")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Usage Type')
    plt.tight_layout()

    # Encoding plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close("all")

    # --- Return Results ---
    
    return jsonify({
        "mse": avg_mse,
        "variance": avg_var,
        "explained_variance": explained_var,
        "image": img_b64,
        "note": note
    })


# Global pound-to-gram conversion factor
LB_TO_GRAM = 453.592

# --- Helper Functions ---

def normalize_text(s):
    """Normalizes text by removing non-alphanumeric chars and converting to lowercase."""
    if isinstance(s, str):
        s = s.strip().lower()
        s = re.sub(r"[^a-z]+$", "", s)
        s = re.sub(r"[^a-z]+", "", s) 
        return s
    return s

# --- Endpoints ---

@app.route("/get_unique_months")
def get_unique_months():
    global item # Assuming 'item' global variable is available

    if item is None or item.empty:
        return jsonify({"months": []})
    
    try:
        # Use a copy and clean month column before finding unique values
        df_temp = item.copy()
        # Ensure 'month' is treated as a string
        months = df_temp['month'].astype(str).str.strip().unique().tolist()
        return jsonify({"months": sorted(months)})
    except KeyError:
        return jsonify({"months": []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/shipment_vs_usage_plot", methods=["POST"])
def shipment_vs_usage_plot():
    global item, ship # Use global item and ship DataFrames

    if item is None or item.empty or ship is None or ship.empty:
        return jsonify({"error": "Item or Ship data not loaded."}), 400

    req = request.json or {}
    selected_month = req.get("month")

    if not selected_month:
        return jsonify({"error": "Month selection is required."}), 400

    # Assuming LB_TO_GRAM is globally defined as 453.592 and normalize_text() is available

    # The conversion map (normalized item ingredient : normalized ship ingredient)
    INGREDIENT_CONVERSION_MAP = {
        "braisedbeefusedg": "beef", 
        "braisedchickeng": "chicken", 
        "eggcount": "egg", 
        "riceg": "rice", 
        "ramencount": "ramen", 
        "ricenoodlesg": "ricenoodles", 
        "chickenthighpcs": "chicken", 
        "chickenwingspcs": "chickenwings", 
        "flourg": "flour", 
        "greenonion": "greenonion", 
        "cilantro": "cilantro", 
        "whiteonion": "whiteonion", 
        "peasg": "peascarrot", 
        "bokchoyg": "bokchoy", 
        "tapiocastarch": "tapiocastarch"
    }

    # Normalized list of target ingredient columns in the item dataset
    ITEM_TARGETS_NORMALIZED = list(INGREDIENT_CONVERSION_MAP.keys())
    
    # Unit conversion flags (1 if conversion from g to lbs is needed, 0 otherwise)
    UNIT_CONVERSION_REQUIRED = {
        "braisedbeefusedg": 1, "braisedchickeng": 1, "riceg": 1, 
        "ricenoodlesg": 1, "flourg": 1, "peasg": 1, "carrotg": 1, # Including carrotg per context
        "bokchoyg": 1, 
    }
    
    # --- 1. Process ITEM (Usage) Data ---

    # Temporarily normalize item columns for access
    item_temp = item.copy()
    item_temp.columns = item_temp.columns.map(normalize_text)
    
    # Filter by month
    month_filter = item_temp['month'].astype(str).str.strip() == selected_month.strip()
    filtered_item_df = item_temp[month_filter]

    if filtered_item_df.empty:
        return jsonify({"error": f"No usage data found for the month: {selected_month}."}), 400

    # Aggregate item ingredients (Usage sums)
    usage_sums = {}
    for item_col in ITEM_TARGETS_NORMALIZED:
        if item_col in filtered_item_df.columns:
            value = pd.to_numeric(filtered_item_df[item_col], errors='coerce').fillna(0).sum()
            
            # Apply unit conversion (grams to pounds) if necessary
            if UNIT_CONVERSION_REQUIRED.get(item_col, 0) == 1 and value != 0:
                value /= 453.592 # Using LB_TO_GRAM constant value
                
            usage_sums[item_col] = value
        # Disregard item columns not present in the filtered dataframe

    # --- 2. Process SHIPMENT Data ---

    ship_temp = ship.copy()
    # Normalize 'Ingredient' column for joining/mapping
    ship_temp['normalized_ingredient'] = ship_temp['Ingredient'].astype(str).apply(normalize_text)

    # Aggregate total shipment for each ingredient listed in the map
    shipment_total = {}
    
    # Reverse the map to easily look up which item columns correspond to a ship ingredient
    ship_to_item_map = {}
    # NEW MAP: Normalized item column -> Original Ship Ingredient Display Name
    item_col_to_display_name = {}

    for k, v in INGREDIENT_CONVERSION_MAP.items():
        ship_to_item_map.setdefault(v, []).append(k)

    # Calculate the shipment total for all mapped ingredients
    for ship_ing_normalized in ship_to_item_map.keys():
        # Find matching rows in ship data
        ship_rows = ship_temp[ship_temp['normalized_ingredient'] == ship_ing_normalized]
        
        if not ship_rows.empty:
            # FIX 1: Corrected column access from 'total' (which doesn't exist) to 'Quantity per month'
            total_shipment_amount = pd.to_numeric(ship_rows['total'], errors='coerce').fillna(0).sum()
            
            # Get the original, capitalized display name (we take the first unique value)
            original_display_name = ship_rows['Ingredient'].iloc[0]
            
            # The shipment amount must be distributed/stored under the corresponding item column name
            for item_col in ship_to_item_map[ship_ing_normalized]:
                shipment_total[item_col] = total_shipment_amount
                # FIX 2: Store the display name against the item column key
                item_col_to_display_name[item_col] = original_display_name
        else:
            # If no shipment found, use the normalized name as a fallback display name
            for item_col in ship_to_item_map[ship_ing_normalized]:
                shipment_total[item_col] = 0
                # Fallback: Capitalize the normalized name for display
                item_col_to_display_name[item_col] = ship_ing_normalized.capitalize()


    # --- 3. Final Data Assembly and Plotting ---

    final_data = []
    
    # Use the display names as the key for the graph
    for item_col in ITEM_TARGETS_NORMALIZED:
        if item_col in usage_sums:
            final_data.append({
                # FIX 3: Use the human-readable display name for the x-axis label
                'Ingredient': item_col_to_display_name.get(item_col, item_col),
                'Used': usage_sums.get(item_col, 0),
                'Shipped': shipment_total.get(item_col, 0)
            })

    if not final_data:
        return jsonify({"error": "No data found after filtering and mapping ingredients."}), 400

    plot_df = pd.DataFrame(final_data).set_index('Ingredient')
    
    # Plotting
    plt.figure(figsize=(15, 8))
    plot_df.plot(kind='bar', ax=plt.gca(), width=0.8)
    
    plt.title(f"Shipment vs. Usage per Ingredient for {selected_month}", fontsize=16)
    plt.ylabel("Amount (in Pounds or Pieces/Counts)", fontsize=12)
    plt.xlabel("Ingredient", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Category')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Encoding plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close("all")

    return jsonify({
        "image": img_b64,
        "note": f"Chart generated for {selected_month}."
    })



if __name__ == "__main__":
    app.run(debug=True, port = 5001)