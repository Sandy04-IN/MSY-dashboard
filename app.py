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
from collections import Counter
from collections import defaultdict


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
LB_TO_GRAM = 453.592
TARGETS_INGREDIENTS = [
    "braisedbeefusedg", "braisedchickeng", "braisedporkg", "eggcount", 
    "riceg", "ramencount", "ricenoodlesg", "chickenthighpcs", 
    "chickenwingspcs", "flourg", "picklecabbage", "greenonion", 
    "cilantro", "whiteonion", "peasg", "carrotg", "bokchoyg", "tapiocastarch"
]
INGREDIENT_DISPLAY_MAP = {
    "braisedbeefusedg": "Braised Beef", 
    "braisedchickeng": "Braised Chicken", 
    "braisedporkg": "Braised Pork", 
    "eggcount": "Egg", 
    "riceg": "Rice", 
    "ramencount": "Ramen", 
    "ricenoodlesg": "Rice Noodles", 
    "chickenthighpcs": "Chicken Thigh", 
    "chickenwingspcs": "Chicken Wings", 
    "flourg": "Flour", 
    "greenonion": "Green Onion", 
    "cilantro": "Cilantro", 
    "whiteonion": "White Onion", 
    "peasg": "Peas",
    "carrotg": "Carrot",
    "bokchoyg": "Bok Choy", 
    "tapiocastarch": "Tapioca Starch",
    "picklecabbage": "Pickled Cabbage"
}







HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Mai KitShan</title>
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
    const { useState, useEffect } = React;
    
    function usePersistentState(key, defaultValue) {
        const [state, setState] = useState(() => {
            try {
            const stored = localStorage.getItem(key);
            return stored ? JSON.parse(stored) : defaultValue;
            } catch {
            return defaultValue;
            }
        });

        useEffect(() => {
            try {
            localStorage.setItem(key, JSON.stringify(state));
            } catch (e) {
            console.error("Failed to save state:", e);
            }
        }, [key, state]);

        return [state, setState];
    }
 
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
            // FIX: Clear previous selections when the grouping category changes
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
                    <option value="Shipment">Shipment</option>
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
        const [monthToPredict, setMonthToPredict] = usePersistentState("nextMonthUsage_monthToPredict", "");
        const [predictionResults, setPredictionResults] = usePersistentState("nextMonthUsage_results", null);
        const [loading, setLoading] = React.useState(false); // no need to persist this
        const [imgUrl, setImgUrl] = usePersistentState("nextMonthUsage_imgUrl", "");
        const [note, setNote] = usePersistentState("nextMonthUsage_note", "");

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
    

    // Cost Prediction Tab
    const CostPrediction = () => {
        const [predictionResults, setPredictionResults] = usePersistentState("costPrediction_results", null);
        const [imgUrl, setImgUrl] = usePersistentState("costPrediction_imgUrl", "");
        const [note, setNote] = usePersistentState("costPrediction_note", "");
        const [coefficientTable, setCoefficientTable] = usePersistentState("costPrediction_coefficientTable", []);
        const [loading, setLoading] = React.useState(false);

        const handleRunPrediction = async () => {
            setLoading(true);
            setPredictionResults(null);
            setImgUrl("");
            setNote("");
            setCoefficientTable([]); // Reset table data

            try {
                const response = await fetch("/predict_cost_loocv", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" }
                });

                if (!response.ok) {
                    const txt = await response.text();
                    setNote("Prediction Error: " + txt);
                    return;
                }

                const data = await response.json();
                
                if (data.error) {
                    setNote("Error: " + data.error);
                    return;
                }
                
                if (data.image) {
                    setImgUrl("data:image/png;base64," + data.image);
                }
                
                setPredictionResults(data);
                setNote(data.note || "Cost prediction completed successfully!");
                setCoefficientTable(data.coefficient_table || []); // SAVE NEW COEFFICIENT DATA

            } catch (e) {
                console.error("Cost prediction error:", e);
                setNote("An unexpected error occurred during prediction.");
            } finally {
                setLoading(false);
            }
        };

        return (
            <div>
                <h2>Cost Prediction</h2>
                <p>
                    Metrics displayed are aggregated across all predicted data points.
                </p>

                <button 
                    onClick={handleRunPrediction} 
                    disabled={loading}
                    style={{ marginTop: 15, marginBottom: 25 }}
                >
                    {loading ? "Training and Predicting..." : "Run Cost Prediction"}
                </button>

                {note && (
                    <div style={{ color: predictionResults?.error ? "red" : "black", marginBottom: 15 }}>
                        <strong>{note}</strong>
                    </div>
                )}

                {predictionResults && (
                    <div style={{ marginTop: 20 }}>
                        <h3>Aggregate Prediction Metrics</h3>
                        {/* ... (Metrics Display remains the same) ... */}
                        <p>
                            <strong>Total Mean Squared Error (MSE):</strong> {typeof predictionResults.mse === "number" ? predictionResults.mse.toFixed(4) : predictionResults.mse || 'N/A'}
                        </p>
                        <p>
                            <strong>Total Variance of Actual Cost:</strong> {typeof predictionResults.variance === "number" ? predictionResults.variance.toFixed(4) : predictionResults.variance || 'N/A'}
                        </p>
                        <p>
                            <strong>Explained Variance (R²):</strong> {typeof predictionResults.explained_variance === "number" ? predictionResults.explained_variance.toFixed(4) : predictionResults.explained_variance || 'N/A'}
                        </p>
                        
                        {imgUrl && (
                            <div style={{ marginTop: 20 }}>
                                <img src={imgUrl} alt="Actual vs Predicted Cost Scatter Plot" style={{ maxWidth: "600px" }} />
                            </div>
                        )}
                    </div>
                )}

                {/* NEW COEFFICIENT TABLE DISPLAY */}
                {coefficientTable.length > 0 && (
                    <div style={{ marginTop: 40 }}>
                        <h3>Ingredients Ranked by Cost Influence</h3>
                        <table style={{ borderCollapse: 'collapse', width: '45%', minWidth: '300px', border: '1px solid #ddd' }}>
                            <thead>
                                <tr style={{ backgroundColor: '#e6f7ff' }}>
                                    <th style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'left' }}>Ingredient</th>
                                    <th style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'right' }}>Coefficient</th>
                                </tr>
                            </thead>
                            <tbody>
                                {coefficientTable.map((row, index) => (
                                    <tr key={index}>
                                        <td style={{ border: '1px solid #ddd', padding: '8px' }}>{row.Ingredient}</td>
                                        <td 
                                            style={{ 
                                                border: '1px solid #ddd', 
                                                padding: '8px', 
                                                textAlign: 'right',
                                                // Optional: Color code based on sign
                                                color: row.Coefficient > 0 ? 'green' : (row.Coefficient < 0 ? 'red' : 'inherit')
                                            }}
                                        >
                                            {row.Coefficient.toFixed(4)}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        );
    };
    
    // Shipment vs. Usage Tab
    const ShipmentVsUsage = () => {
        const [selectedMonth, setSelectedMonth] = usePersistentState("shipmentVsUsage_selectedMonth", "");
        const [imgUrl, setImgUrl] = usePersistentState("shipmentVsUsage_imgUrl", "");
        const [note, setNote] = usePersistentState("shipmentVsUsage_note", "");
        const [tableData, setTableData] = usePersistentState("shipmentVsUsage_tableData", []);
        const [actionTableData, setActionTableData] = usePersistentState("shipmentVsUsage_actionTableData", []);
        const [monthOptions, setMonthOptions] = React.useState([]);
        const [loading, setLoading] = React.useState(false);

        // Fetch unique months for the dropdown on component load (unchanged)
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
            setTableData([]); // Reset table data
            setActionTableData([]); // Make sure to reset this as well

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
                    setTableData(data.table_data || []); 
                    setActionTableData(data.action_table_data || []); // <--- THIS LINE WAS MISSING AND IS THE FIX
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
                <p>There is a large difference between the "Used" and "Shipped" values due to lack of ingredient data for most menu items.</p>

                {/* ... (Month selection and button: unchanged) ... */}

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

                {/* NEW TABLE DISPLAY */}
                {tableData.length > 0 && (
                    <div style={{ marginTop: 30 }}>
                        <h3>Data Table ({selectedMonth})</h3>
                        <table style={{ borderCollapse: 'collapse', width: '100%', border: '1px solid #ddd' }}>
                            <thead>
                                <tr style={{ backgroundColor: '#f2f2f2' }}>
                                    <th style={{ border: '1px solid #ddd', padding: '8px' }}>Ingredient</th>
                                    <th style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'center' }}>Units</th>
                                    <th style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'right' }}>Used</th>
                                    <th style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'right' }}>Shipped</th>
                                </tr>
                            </thead>
                            <tbody>
                                {tableData.map((row, index) => (
                                    <tr key={index}>
                                        <td style={{ border: '1px solid #ddd', padding: '8px' }}>{row.Ingredient}</td>
                                        <td style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'center' }}>{row.Units}</td>
                                        <td style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'right' }}>{row.Used}</td>
                                        <td style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'right' }}>{row.Shipped}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
                {actionTableData.length > 0 && (
                    <div style={{ marginTop: 40, width: '100%' }}>
                        <h3>Inventory Action Recommendations </h3>
                        <table style={{ borderCollapse: 'collapse', width: '100%', border: '1px solid #ddd' }}>
                            <thead>
                                <tr style={{ backgroundColor: '#fff8e1' }}>
                                    <th style={{ border: '1px solid #ddd', padding: '8px', width: '50%' }}>Order Less</th>
                                    <th style={{ border: '1px solid #ddd', padding: '8px', width: '50%' }}>Order More</th>
                                </tr>
                            </thead>
                            <tbody>
                                {actionTableData.map((row, index) => (
                                    <tr key={index}>
                                        <td style={{ border: '1px solid #ddd', padding: '8px', color: row['Order Less'] ? 'black' : 'inherit' }}>{row['Order Less']}</td>
                                        <td style={{ border: '1px solid #ddd', padding: '8px', color: row['Order More'] ? 'black' : 'inherit' }}>{row['Order More']}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        );
    };
    
    
    
    // Used/Shipped Timeline Tab
    const UsedShippedTimeline = () => {
        const [selectedIngredient, setSelectedIngredient] = usePersistentState("usedShippedTimeline_selectedIngredient", "");
        const [imgUrl, setImgUrl] = usePersistentState("usedShippedTimeline_imgUrl", "");
        const [note, setNote] = usePersistentState("usedShippedTimeline_note", "");
        const [tableData, setTableData] = usePersistentState("usedShippedTimeline_tableData", []);
        const [ingredientOptions, setIngredientOptions] = React.useState([]);
        const [loading, setLoading] = React.useState(false);

        // Fetch ingredients on mount
        React.useEffect(() => {
            fetch("/get_ingredient_list")
                .then((r) => r.json())
                .then((data) => setIngredientOptions(data.ingredients || []))
                .catch((e) => console.error("Error fetching ingredients:", e));
        }, []);

        const handleGenerateChart = async () => {
            if (!selectedIngredient) {
                setNote("Please select an ingredient.");
                return;
            }

            setLoading(true);
            setImgUrl("");
            setNote("");
            setTableData([]);

            try {
                const response = await fetch("/used_shipped_timeline_plot", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ ingredient: selectedIngredient }),
                });

                const data = await response.json();

                if (!response.ok || data.error) {
                    setNote("Error: " + (data.error || "Failed to generate chart."));
                    return;
                }

                if (data.image) {
                    setImgUrl("data:image/png;base64," + data.image);
                    setNote(data.note || "Timeline generated successfully.");
                    setTableData(data.table_data || []);
                }
            } catch (e) {
                console.error("Plotting error:", e);
                setNote("An unexpected error occurred.");
            } finally {
                setLoading(false);
            }
        };

        return (
            <div>
                <h2>Used/Shipped Timeline</h2>
                <p>View how usage and shipment quantities change over time for each ingredient.</p>

                <div style={{ marginBottom: 15 }}>
                    <label>Ingredient: </label>
                    <select
                        value={selectedIngredient}
                        onChange={(e) => setSelectedIngredient(e.target.value)}
                    >
                        <option value="">--Select Ingredient--</option>
                        {ingredientOptions.map((ing, i) => (
                            <option key={i} value={ing}>
                                {ing}
                            </option>
                        ))}
                    </select>

                    <button
                        onClick={handleGenerateChart}
                        disabled={loading || !selectedIngredient}
                        style={{ marginLeft: 10 }}
                    >
                        {loading ? "Generating..." : "Generate Timeline"}
                    </button>
                </div>

                {note && (
                    <div style={{ color: "darkblue", marginBottom: 15 }}>
                        <strong>{note}</strong>
                    </div>
                )}

                {imgUrl && (
                    <div style={{ marginTop: 20 }}>
                        <img src={imgUrl} alt="Used vs Shipped Timeline" style={{ maxWidth: "100%" }} />
                    </div>
                )}

                {tableData.length > 0 && (
                    <div style={{ marginTop: 30 }}>
                        <h3>Data Table ({selectedIngredient})</h3>
                        <table style={{ borderCollapse: "collapse", width: "100%", border: "1px solid #ddd" }}>
                            <thead>
                                <tr style={{ backgroundColor: "#f2f2f2" }}>
                                    <th style={{ border: "1px solid #ddd", padding: "8px" }}>Month</th>
                                    <th style={{ border: "1px solid #ddd", padding: "8px", textAlign: "right" }}>Used</th>
                                    <th style={{ border: "1px solid #ddd", padding: "8px", textAlign: "right" }}>Shipped</th>
                                </tr>
                            </thead>
                            <tbody>
                                {tableData.map((row, index) => (
                                    <tr key={index}>
                                        <td style={{ border: "1px solid #ddd", padding: "8px" }}>{row.month}</td>
                                        <td style={{ border: "1px solid #ddd", padding: "8px", textAlign: "right" }}>{row.Used}</td>
                                        <td style={{ border: "1px solid #ddd", padding: "8px", textAlign: "right" }}>{row.Shipped}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        );
    };
    
    // Bestsellers Tab
    const Bestsellers = () => {
        const [selectedMonth, setSelectedMonth] = usePersistentState("bestsellers_selectedMonth", "");
        const [imgUrl, setImgUrl] = usePersistentState("bestsellers_imgUrl", "");
        const [note, setNote] = usePersistentState("bestsellers_note", "");
        const [ingredientTable, setIngredientTable] = usePersistentState("bestsellers_ingredientTable", []);
        const [frequencyTable, setFrequencyTable] = usePersistentState("bestsellers_frequencyTable", []);
        const [monthOptions, setMonthOptions] = React.useState([]);
        const [loading, setLoading] = React.useState(false);

        // --- Fetch unique months on load (unchanged) ---
        React.useEffect(() => {
            fetch("/get_unique_months")
                .then((r) => r.json())
                .then((data) => {
                    setMonthOptions(data.months || []);
                })
                .catch((e) => console.error("Error fetching months:", e));
        }, []);

        const handleGenerateChart = async () => {
            setLoading(true);
            setImgUrl("");
            setNote("");
            setIngredientTable([]);
            setFrequencyTable([]); // <-- Reset frequency table

            try {
                const response = await fetch("/bestsellers_plot", {
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
                    setIngredientTable(data.ingredient_table || []); 
                    setFrequencyTable(data.frequency_table || []); // <-- SAVE NEW FREQUENCY DATA
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
                <h2>Bestsellers Analysis</h2>
                
                {/* ... (Month selection and chart display remain the same) ... */}

                <div style={{ marginBottom: 15 }}>
                    <label>Month: </label>
                    <select 
                        value={selectedMonth} 
                        onChange={(e) => setSelectedMonth(e.target.value)}
                    >
                        <option value="">--All Months--</option> {/* Optional value */}
                        {monthOptions.map((month) => (
                            <option key={month} value={month}>
                                {month}
                            </option>
                        ))}
                    </select>

                    <button 
                        onClick={handleGenerateChart} 
                        disabled={loading}
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
                        <img src={imgUrl} alt="Top 10 Bestsellers Plot" style={{ maxWidth: "100%" }} />
                    </div>
                )}

                {/* Existing Ingredient Breakdown Table */}
                {ingredientTable.length > 0 && (
                    <div style={{ marginTop: 40, width: '100%' }}>
                        <h3>Ingredient Breakdown for Top Sellers</h3>
                        <table style={{ borderCollapse: 'collapse', width: '100%', border: '1px solid #ddd' }}>
                            <thead>
                                <tr style={{ backgroundColor: '#f2f2f2' }}>
                                    <th style={{ border: '1px solid #ddd', padding: '8px', width: '30%' }}>Food Item</th>
                                    <th style={{ border: '1px solid #ddd', padding: '8px', width: '70%' }}>Ingredients Used</th>
                                </tr>
                            </thead>
                            <tbody>
                                {ingredientTable.map((row, index) => (
                                    <tr key={index}>
                                        <td style={{ border: '1px solid #ddd', padding: '8px' }}>{row['Food Item']}</td>
                                        <td style={{ border: '1px solid #ddd', padding: '8px' }}>
                                            {row.Ingredients.join(', ')}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
                
                {/* NEW: Ingredient Frequency Table */}
                {frequencyTable.length > 0 && (
                    <div style={{ marginTop: 40, width: '100%' }}>
                        <h3>Ingredient Frequency in Top Sellers</h3>
                        <table style={{ borderCollapse: 'collapse', width: '35%', border: '1px solid #ddd' }}>
                            <thead>
                                <tr style={{ backgroundColor: '#e6e6fa' }}>
                                    <th style={{ border: '1px solid #ddd', padding: '8px' }}>Ingredient</th>
                                    <th style={{ border: '1px solid #ddd', padding: '8px' }}>Frequency</th>
                                </tr>
                            </thead>
                            <tbody>
                                {frequencyTable.map((row, index) => (
                                    <tr key={index}>
                                        <td style={{ border: '1px solid #ddd', padding: '8px' }}>{row.Ingredient}</td>
                                        <td style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'center' }}>{row.Frequency}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
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
              <button onClick={() => setActiveTab("CostPrediction")}>Cost Prediction</button>
              <button onClick={() => setActiveTab("ShipmentVsUsage")}>Shipment vs. Usage</button>
              <button onClick={() => setActiveTab("UsedShippedTimeline")}>Used/Shipped Timeline</button>
              <button onClick={() => setActiveTab("Bestsellers")}>Bestsellers</button>
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
            {activeTab === "CostPrediction" && (
              <CostPrediction columnNames={columnNames} modelBuilt={modelBuilt} modelTargets={modelTargets} />
            )}
            {activeTab === "ShipmentVsUsage" && (
                <ShipmentVsUsage columnNames={columnNames} modelBuilt={modelBuilt} modelTargets={modelTargets} />
            )}
            {activeTab === "UsedShippedTimeline" && (
                <UsedShippedTimeline columnNames={columnNames} modelBuilt={modelBuilt} modelTargets={modelTargets} />
            )}
            {activeTab === "Bestsellers" && (
                <Bestsellers columnNames={columnNames} modelBuilt={modelBuilt} modelTargets={modelTargets} />
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
        for row in ship.itertuples():
            freq = str(row.frequency).strip().lower()
            if freq == "weekly":
                ship.at[row.Index, "total"] *= 4  # Approximate 4 weeks in a month
            elif freq == "biweekly":
                ship.at[row.Index, "total"] *= 2  # Approximate 2 biweeks in a month

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
    global group, category, item, ship
    req = request.json or {}
    groupBy = req.get("groupBy")

    df_map = {
        "Group": group,
        "Category": category,
        "Item": item,
        "Shipment": ship
    }

    df = df_map.get(groupBy)
    if df is None:
        return jsonify({"columns": []})

    columns = list(df.columns)
    return jsonify({"columns": columns})



@app.route("/plot", methods=["POST"])
def plot():
    global group, category, item, ship
    req = request.json or {}
    x = req.get("x")
    y = req.get("y")
    groupBy = req.get("groupBy")
    plot_type = req.get("plotType", "scatterplot").lower()

    df_map = {
        "Group": group,
        "Category": category,
        "Item": item,
        "Shipment": ship
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
    

    col_name_map_forward = {col: normalize_text(col) for col in item.columns}
    
    col_name_map_reverse = {v: k for k, v in col_name_map_forward.items()}
    
    df_pred_base = item.copy()
    df_pred_base.columns = df_pred_base.columns.map(col_name_map_forward)
    
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
    
    # --- FIX: Rename Index to Unnormalized Names ---
    
    # Create the mapping: Original Name -> Normalized Name
    col_name_map_forward = {col: normalize_text(col) for col in item.columns}
    
    # Create the REVERSE mapping: Normalized Name -> Original Name (for plotting)
    # This is the key to using the unnormalized names for plotting later.
    col_name_map_reverse = {v: k for k, v in col_name_map_forward.items()}

    # 1. Create a dictionary to map normalized targets (the current index) back to their original names
    # Only map the targets we actually processed
    plot_label_map = {
        norm_name: col_name_map_reverse.get(norm_name, norm_name)
        for norm_name in plot_data.index
    }
    
    # 2. Apply the mapping to the DataFrame index
    plot_data.index = plot_data.index.map(plot_label_map)
    
    # --- End Fix ---
    
    # Ensure import is inside function if it wasn't global
    import matplotlib.pyplot as plt 
    import io, base64 
    
    plt.figure(figsize=(12, 6))
    
    # plot_data.plot uses the index for X-axis labels (now unnormalized)
    plot_data.plot(kind='bar', ax=plt.gca(), width=0.8)
    
    plt.title(f"Actual vs. Predicted Ingredient Usage (Total for {month_to_predict_str})")
    plt.ylabel("Total Usage (Sum)")
    plt.xlabel("Ingredient")
    plt.xticks(rotation=45, ha='right') # Ticks will use the unnormalized names
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





@app.route("/predict_cost_loocv", methods=["POST"])
def predict_cost_loocv():
    global item, TARGETS_INGREDIENTS # Included TARGETS_INGREDIENTS for clear global access (assuming it's defined outside)
    
    if item is None or item.empty:
        return jsonify({"error": "Item data not loaded."}), 400

    # ... (omitted sections 1-3 which prepare df_agg, X_full, y_full, etc.) ...
    
    df_temp = item.copy()
    col_name_map_forward = {col: normalize_text(col) for col in item.columns}
    df_pred_base = item.copy()
    df_pred_base.columns = df_pred_base.columns.map(col_name_map_forward)
    df_pred_base['cost'] = pd.to_numeric(df_pred_base.get('cost'), errors='coerce').fillna(0)
    df_pred_base = df_pred_base.dropna(subset=['cost'])

    ingredient_cols = [col for col in TARGETS_INGREDIENTS if col in df_pred_base.columns]
    
    if not ingredient_cols:
        return jsonify({"error": "Ingredient columns not found in dataset after normalization."}), 400
    
    df_pred_base['total_ing_usage'] = df_pred_base[ingredient_cols].sum(axis=1)
    df_filtered = df_pred_base[df_pred_base['total_ing_usage'] > 0].copy()
    
    if df_filtered.empty:
        return jsonify({"error": "No relevant data (cost > 0 AND ingredient usage > 0) found for prediction."}), 400

    food_item_col = 'itemname'
    if food_item_col not in df_filtered.columns:
        return jsonify({"error": "Required column 'itemname' not found."}), 400
        
    cols_to_aggregate = ['cost'] + ingredient_cols
    df_agg = df_filtered.groupby(food_item_col)[cols_to_aggregate].mean().reset_index()

    X_full = df_agg[ingredient_cols].copy()
    y_full = df_agg['cost'].copy()
    
    X_means = X_full.mean().to_dict()
    X_full = X_full.fillna(X_means)
    
    unique_items = df_agg[food_item_col].unique()
    final_variance_base = np.var(y_full)

    # --- 2. LOOCV-Style Prediction and Metric Aggregation ---
    
    all_actuals_agg = []
    all_predictions_agg = []
    food_item_labels = []
    item_mse_list = []
    all_model_coefficients = defaultdict(list)
    
    for item_index, item_value in enumerate(unique_items):
        test_mask = df_agg[food_item_col] == item_value
        
        # NOTE: X_train and X_test are COPIED from X_full here
        X_train = X_full[~test_mask].copy()
        y_train = y_full[~test_mask].copy()
        X_test = X_full[test_mask].copy()
        y_test = y_full[test_mask].copy() 

        if X_train.empty or X_test.empty or y_test.empty:
            continue
            
        stable_predictors = [col for col in ingredient_cols if X_train[col].nunique() > 1]
        
        if not stable_predictors:
            mean_pred = y_train.mean()
            all_actuals_agg.append(y_test.iloc[0])
            all_predictions_agg.append(mean_pred)
            food_item_labels.append(item_value)
            
            # Store 0 for coefficients if no features selected
            for feature in ingredient_cols: # Use all ingredient_cols for coefficient accumulation
                all_model_coefficients[feature].append(0.0)
            continue
            
        # -----------------------------------------------------------
        # FEATURE STANDARDIZATION (FIX IMPLEMENTATION)
        # -----------------------------------------------------------
        scaler = StandardScaler()
        
        # 1. Fit scaler only on the training features that are stable
        X_train_stable = X_train[stable_predictors]
        scaler.fit(X_train_stable)

        # 2. Transform both the training and test sets, maintaining structure
        X_train_scaled_array = scaler.transform(X_train_stable)
        X_test_scaled_array = scaler.transform(X_test[stable_predictors])
        
        # Convert back to DataFrame for sm.OLS compatibility
        X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=stable_predictors, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=stable_predictors, index=X_test.index)
        
        # --- Stepwise Selection (Both Directions, AIC) ---
        selected_features = []
        remaining = list(stable_predictors)
        best_score = float("inf")
        
        while True:
            scores_to_add = []
            scores_to_remove = []
            
            # Forward Step uses X_train_scaled
            for candidate in remaining:
                try:
                    features = selected_features + [candidate]
                    model_try = sm.OLS(y_train, sm.add_constant(X_train_scaled[features], has_constant="add")).fit()
                    scores_to_add.append((model_try.aic, candidate))
                except (np.linalg.LinAlgError, Exception): continue

            # Backward Step uses X_train_scaled
            if selected_features:
                for candidate in selected_features:
                    try:
                        features = [f for f in selected_features if f != candidate]
                        if not features:
                            # Model with only constant
                            model_try = sm.OLS(y_train, sm.add_constant(pd.DataFrame(index=X_train.index), has_constant="add")).fit()
                        else:
                            model_try = sm.OLS(y_train, sm.add_constant(X_train_scaled[features], has_constant="add")).fit()
                        scores_to_remove.append((model_try.aic, candidate))
                    except (np.linalg.LinAlgError, Exception): continue

            best_forward = min(scores_to_add) if scores_to_add else (float("inf"), None)
            best_backward = min(scores_to_remove) if scores_to_remove else (float("inf"), None)
            current_best_score = min(best_forward[0], best_backward[0])
            
            if current_best_score < best_score:
                if current_best_score == best_forward[0]:
                    best_candidate = best_forward[1]
                    selected_features.append(best_candidate)
                    if best_candidate in remaining: remaining.remove(best_candidate)
                else:
                    best_candidate = best_backward[1]
                    selected_features.remove(best_candidate)
                    remaining.append(best_candidate)
                best_score = current_best_score
            else:
                break
        
        # --- Final Prediction and Metric Extraction ---
        
        predicted_cost = y_train.mean() 
        
        if selected_features:
            try:
                final_model = sm.OLS(y_train, sm.add_constant(X_train_scaled[selected_features], has_constant="add")).fit()
                
                # STORE COEFFICIENTS (NEW): Store coefficients for ALL ingredient columns
                for feature in ingredient_cols: 
                    # Store the coefficient if selected, otherwise 0.0
                    if feature in final_model.params:
                        all_model_coefficients[feature].append(final_model.params[feature])
                    else:
                        all_model_coefficients[feature].append(0.0)
                
                # Prepare test data with constant and feature alignment
                X_test_aligned = sm.add_constant(X_test_scaled[selected_features], has_constant="add")
                X_test_aligned = X_test_aligned.reindex(columns=final_model.model.exog_names, fill_value=0.0).astype(float)
                
                preds = final_model.predict(X_test_aligned)
                predicted_cost = preds.iloc[0] 
                
            except Exception:
                predicted_cost = y_train.mean() 
                # Store 0 for coefficients if model failed
                for feature in ingredient_cols:
                    all_model_coefficients[feature].append(0.0)
        else:
             # Store 0 for coefficients if no features selected
             for feature in ingredient_cols:
                 all_model_coefficients[feature].append(0.0)

        actual_cost = y_test.iloc[0]
        
        all_actuals_agg.append(actual_cost)
        all_predictions_agg.append(predicted_cost)
        food_item_labels.append(item_value)
        
        item_mse = mean_squared_error([actual_cost], [predicted_cost])
        item_mse_list.append({'Item': item_value, 'MSE': item_mse})

    # ... (sections 3, 4, 5, and 6 remain the same) ...
    # This ensures that the final aggregate metrics, plotting, and coefficient table creation
    # use the data generated by the standardized models.
    
    # Convert lists to NumPy arrays for final metric calculation
    actual_array = np.array(all_actuals_agg)
    pred_array = np.array(all_predictions_agg)
    
    if actual_array.size == 0:
        return jsonify({"error": "Prediction completed successfully, but no valid data points were produced for metrics."}), 400

    # --- 3. Aggregate Metrics ---
    final_mse = mean_squared_error(actual_array, pred_array)
    final_variance = final_variance_base
    final_explained_variance = 1 - (final_mse / final_variance) if final_variance != 0 else 0.0

    # 4a. Average Coefficients Table (NEW)
    avg_coefficients = {}
    for feature in ingredient_cols:
        avg_coefficients[feature] = np.mean(all_model_coefficients[feature])
        
    coefficient_table = []
    for norm_name, avg_coeff in avg_coefficients.items():
        # Look up display name (e.g., braisedbeefusedg -> Braised Beef)
        display_name = INGREDIENT_DISPLAY_MAP.get(norm_name, norm_name.replace('g', '').capitalize())
        coefficient_table.append({
            'Ingredient': display_name,
            'Coefficient': avg_coeff
        })

    # Sort coefficient table by Coefficient value (descending)
    coefficient_table.sort(key=lambda x: x['Coefficient'], reverse=True)

    # 4b. Plot Data (Side-by-Side Bar Chart sorted by Item MSE)
    
    # Combine item data with MSE
    plot_data_full = pd.DataFrame({
        'Food Item': food_item_labels,
        'Actual Cost': actual_array,
        'Predicted Cost': pred_array,
        'Item_MSE': [d['MSE'] for d in item_mse_list]
    })
    
    # Sort the items by their individual MSE (ASCENDING)
    plot_data_full = plot_data_full.sort_values(by='Item_MSE', ascending=False)
    item_order_by_mse = plot_data_full['Food Item'].tolist()
    
    # Reshape for Seaborn plotting
    plot_df = pd.melt(plot_data_full, id_vars=['Food Item'], value_vars=['Actual Cost', 'Predicted Cost'], 
                      var_name='Cost Type', value_name='Cost Value')
    
    # --- 5. Plotting (Side-by-Side Bar Chart sorted by Item MSE) ---
    
    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=plot_df, 
        x='Food Item', 
        y='Cost Value', 
        hue='Cost Type', 
        order=item_order_by_mse, # <-- SORT BY MSE
    )
    
    plt.title("Actual vs. Predicted Cost by Food Item", fontsize=14)
    plt.xlabel("Food Item", fontsize=12)
    plt.ylabel("Cost", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Cost Type')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Encoding plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close("all")

    # --- 6. Final Response ---
    
    return jsonify({
        "mse": final_mse,
        "variance": final_variance,
        "explained_variance": final_explained_variance,
        "image": img_b64,
        "note": "Cost prediction completed",
        "coefficient_table": coefficient_table # <-- NEW TABLE DATA
    })
    

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
    global item

    if item is None or item.empty:
        return jsonify({"months": []})
    
    try:
        df_temp = item.copy()
        
        # 1. Ensure required columns exist
        if 'month' not in df_temp.columns:
            return jsonify({"error": "Item dataset missing 'month' column."}), 400
        if 'month numerical' not in df_temp.columns:
            return jsonify({"error": "Item dataset missing 'month numerical' column."}), 400
        month_data = df_temp[['month', 'month numerical']].drop_duplicates().copy()
        
        month_data['month numerical'] = pd.to_numeric(month_data['month numerical'], errors='coerce')
        month_data = month_data.dropna(subset=['month numerical'])
        
        # Sort chronologically
        month_data = month_data.sort_values(by='month numerical', ascending=True)
        
        ordered_months = month_data['month'].astype(str).str.strip().tolist()

        return jsonify({"months": ordered_months})
        
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve and sort months: {str(e)}"}), 500




# ... (omitting setup and sections 1-2 for brevity) ...

@app.route("/shipment_vs_usage_plot", methods=["POST"])
def shipment_vs_usage_plot():
    global item, ship 

    if item is None or item.empty or ship is None or ship.empty:
        return jsonify({"error": "Item or Ship data not loaded."}), 400

    req = request.json or {}
    selected_month = req.get("month")

    if not selected_month:
        return jsonify({"error": "Month selection is required."}), 400

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
    ITEM_TARGETS_NORMALIZED = list(INGREDIENT_CONVERSION_MAP.keys())
    UNIT_CONVERSION_REQUIRED = {
        "braisedbeefusedg": 1, "braisedchickeng": 1, "riceg": 1, 
        "ricenoodlesg": 1, "flourg": 1, "peasg": 1, "carrotg": 1, 
        "bokchoyg": 1, "greenonion":1, "cilantro":1,
    }
    LB_TO_GRAM = 453.592

    # Helper function needed inside the route for mapping/processing (assuming it was defined externally)
    def normalize_text(s):
        s = str(s).strip().lower()
        s = re.sub(r"[^a-z]+$", "", s)
        s = re.sub(r"[^a-z]+", "", s)
        return s

    # --- 1. Compute usage per ingredient (Replicating logic to define necessary variables) ---
    item_temp = item.copy()
    item_temp.columns = item_temp.columns.map(normalize_text)
    month_filter = item_temp['month'].astype(str).str.strip() == selected_month.strip()
    filtered_item_df = item_temp[month_filter]
    
    usage_sums = {}
    item_col_to_display_name = {}

    for item_col in ITEM_TARGETS_NORMALIZED:
        if item_col in filtered_item_df.columns:
            # Assuming 'count' is available and numeric
            count_multiplier = pd.to_numeric(filtered_item_df['count'], errors='coerce').fillna(1)
            raw_usage = pd.to_numeric(filtered_item_df[item_col], errors='coerce').fillna(0)
            value = (raw_usage * count_multiplier).sum()

            if UNIT_CONVERSION_REQUIRED.get(item_col, 0) == 1 and value != 0:
                value /= LB_TO_GRAM

            usage_sums[item_col] = value
            mapped_ship_ingredient = INGREDIENT_CONVERSION_MAP.get(item_col, item_col)
            item_col_to_display_name[item_col] = mapped_ship_ingredient.capitalize()
    
    # --- 2. Process SHIPMENT Data (Replicating logic) ---
    ship_temp = ship.copy()
    ship_temp['normalized_ingredient'] = ship_temp['Ingredient'].astype(str).apply(normalize_text)
    shipment_total = {}
    ship_to_item_map = {}
    for k, v in INGREDIENT_CONVERSION_MAP.items():
        ship_to_item_map.setdefault(v, []).append(k)

    for ship_ing_normalized in ship_to_item_map.keys():
        ship_rows = ship_temp[ship_temp['normalized_ingredient'] == ship_ing_normalized]
        
        if not ship_rows.empty:
            # Assuming 'total' is the correct column name here
            total_shipment_amount = pd.to_numeric(ship_rows['total'], errors='coerce').fillna(0).sum()
            original_display_name = ship_rows['Ingredient'].iloc[0]
            
            for item_col in ship_to_item_map[ship_ing_normalized]:
                shipment_total[item_col] = total_shipment_amount
                item_col_to_display_name[item_col] = original_display_name
        else:
            for item_col in ship_to_item_map[ship_ing_normalized]:
                shipment_total[item_col] = 0
                item_col_to_display_name[item_col] = ship_ing_normalized.capitalize()

    # --- 3. Final Data Assembly ---

    final_data = []
    order_less_list = []
    order_more_list = []

    for item_col in ITEM_TARGETS_NORMALIZED:
        used_val = usage_sums.get(item_col, 0)
        shipped_val = shipment_total.get(item_col, 0)
        display_name = item_col_to_display_name.get(item_col, item_col)

        if item_col in usage_sums:
            final_data.append({
                'Ingredient': display_name,
                'Units': 'Pounds' if UNIT_CONVERSION_REQUIRED.get(item_col, 0) == 1 else 'Pieces',
                'Used': used_val,
                'Shipped': shipped_val
            })
            
            # --- Inventory Action Logic (NEW) ---
            if shipped_val > 0:
                # Assuming the original request meant less than 75% for Order Less
                if used_val < (0.75 * shipped_val):
                    order_less_list.append(display_name)
                elif used_val > shipped_val:
                    order_more_list.append(display_name)


    if not final_data:
        return jsonify({"error": "No data found after filtering and mapping ingredients."}), 400

    plot_df = pd.DataFrame(final_data).set_index('Ingredient')
    
    # 4. PREPARE TABULAR DATA 
    table_df = plot_df.reset_index()
    unit_lookup = {d['Ingredient']: d['Units'] for d in final_data}
    table_df['Units'] = table_df['Ingredient'].map(unit_lookup)
    
    table_df['Used'] = table_df['Used'].round(2)
    table_df['Shipped'] = table_df['Shipped'].round(2)
    
    table_data = table_df[['Ingredient', 'Units', 'Used', 'Shipped']].to_dict('records')


    # Prepare the Order Table data structure
    max_len = max(len(order_less_list), len(order_more_list))
    order_less_padded = order_less_list + [''] * (max_len - len(order_less_list))
    order_more_padded = order_more_list + [''] * (max_len - len(order_more_list))
    
    action_table_data = [
        {'Order Less': ol, 'Order More': om} 
        for ol, om in zip(order_less_padded, order_more_padded)
    ]


    # 5. Plotting (With Sorting Applied)
    plt.figure(figsize=(15, 8))
    
    # --- SORTING FIX ---
    # 1. Sort the plotting DataFrame by 'Used' in descending order
    plot_df_sorted = plot_df.sort_values(by='Used', ascending=False)
    
    # 2. Get the new order of index labels
    order_list = plot_df_sorted.index.tolist()

    # 3. Plot the sorted DataFrame
    # Note: plot_df.plot() uses the index order automatically, 
    # but we need to explicitly plot the sorted DataFrame.
    plot_df_sorted.plot(kind='bar', ax=plt.gca(), width=0.8)
    
    plt.title(f"Shipment vs. Usage per Ingredient for {selected_month} (Sorted by Used)", fontsize=16)
    plt.ylabel("Amount (in Pounds or Pieces/Counts)", fontsize=12)
    plt.xlabel("Ingredient", fontsize=12)
    # The tick labels will automatically follow the sorted index
    plt.xticks(rotation=45, ha='right') 
    plt.legend(title='Category')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Encoding plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close("all")

    # 6. RETURN TABULAR DATA
    return jsonify({
        "image": img_b64,
        "note": f"Chart generated for {selected_month}.",
        "table_data": table_data,
        "action_table_data": action_table_data
    })


@app.route("/get_ingredient_list")
def get_ingredient_list():
    """Return unique Ingredient names from the global 'ship' DataFrame."""
    global ship
    if ship is None or ship.empty:
        return jsonify({"ingredients": []})
    try:
        ingredients = ship["Ingredient"].dropna().astype(str).unique().tolist()
        return jsonify({"ingredients": sorted(ingredients)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/used_shipped_timeline_plot", methods=["POST"])
def used_shipped_timeline_plot():
    import re, io, base64, traceback
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np 
    from flask import jsonify

    global item, ship

    try:
        # --- Validation ---
        if item is None or item.empty or ship is None or ship.empty:
            raise ValueError("Item or Ship data not loaded.")

        req = request.json or {}
        selected_ingredient = req.get("ingredient")
        if not selected_ingredient:
            raise ValueError("Ingredient selection is required.")
            
        # --- Normalize text helper ---
        def normalize_text(s):
            s = str(s).strip().lower()
            s = re.sub(r"[^a-z]+$", "", s)
            s = re.sub(r"[^a-z]+", "", s)
            return s

        # Find the correct column name for the total shipment amount
        # Check for 'total' (as shown in your output) or the common alternative 'Quantity per month'
        SHIPMENT_AMOUNT_COL = None
        if 'total' in ship.columns:
            SHIPMENT_AMOUNT_COL = 'total'
        elif 'Quantity per month' in ship.columns:
            SHIPMENT_AMOUNT_COL = 'Quantity per month'
        else:
             raise KeyError("Ship dataset missing the required total shipment column ('total' or 'Quantity per month'). Cannot compute total shipped.")

        # --- Conversion and mapping (omitted for brevity, remains the same) ---
        INGREDIENT_CONVERSION_MAP = {
            "braisedbeefusedg": "beef", "braisedchickeng": "chicken", "eggcount": "egg",
            "riceg": "rice", "ramencount": "ramen", "ricenoodlesg": "ricenoodles", 
            "chickenthighpcs": "chicken", "chickenwingspcs": "chickenwings",
            "flourg": "flour", "greenonion": "greenonion", "cilantro": "cilantro",
            "whiteonion": "whiteonion", "peasg": "peascarrot", "bokchoyg": "bokchoy", 
            "tapiocastarch": "tapiocastarch"
        }

        UNIT_CONVERSION_REQUIRED = {
            "braisedbeefusedg": 1, "braisedchickeng": 1, "riceg": 1,
            "ricenoodlesg": 1, "flourg": 1, "peasg": 1, "carrotg": 1,
            "bokchoyg": 1, "greenonion": 1, "cilantro": 1,
        }
        LB_TO_GRAM = 453.592

        normalized_target = normalize_text(selected_ingredient)

        # Reverse mapping: ship ingredient -> item columns
        ship_to_item_map = {}
        for k, v in INGREDIENT_CONVERSION_MAP.items():
            ship_to_item_map.setdefault(v, []).append(k)

        # --- 1. Calculate Monthly Shipped Total (Shipped value is constant per month) ---
        ship_temp = ship.copy()
        if "Ingredient" not in ship_temp.columns:
             raise KeyError("Ship dataset is missing the 'Ingredient' column.")
            

        ship_temp["normalized_ingredient"] = ship_temp["Ingredient"].astype(str).apply(normalize_text)
        filtered_ship = ship_temp[ship_temp["normalized_ingredient"] == normalized_target]
        
        # Calculate the single monthly total shipment for the ingredient
        monthly_shipment_total = pd.to_numeric(filtered_ship[SHIPMENT_AMOUNT_COL], errors='coerce').fillna(0).sum()
        
        # --- 2. Compute "Used" values per month from item data ---
        item_temp = item.copy()
        item_temp.columns = item_temp.columns.map(normalize_text)

        if "month" not in item_temp.columns or "monthnumerical" not in item_temp.columns:
             raise KeyError("Item dataset missing 'month' or 'monthnumerical' column.")

        used_per_month = []
        related_item_cols = ship_to_item_map.get(normalized_target, [])

        if not related_item_cols:
             raise ValueError(f"No matching columns found in item dataset for ingredient '{selected_ingredient}'.")

        for name, group in item_temp.groupby(["month", "monthnumerical"]):
            month_name, month_num = name
            used_sum = 0
            for col in related_item_cols:
                if col in group.columns:
                    val = (pd.to_numeric(group[col], errors="coerce").fillna(0)) * group["count"]
                    val = val.sum()
                    if UNIT_CONVERSION_REQUIRED.get(col, 0) == 1:
                        val /= LB_TO_GRAM
                    used_sum += val
            used_per_month.append({"month": month_name, "monthnumerical": month_num, "Used": used_sum})

        used_df = pd.DataFrame(used_per_month)
        
        # --- 3. Final Assembly and Plotting Data (Replaced Merge) ---
        
        # Add the constant monthly shipment total to every row of the used data
        used_df['Shipped'] = monthly_shipment_total
        
        # Sort by month numerical for correct timeline plot order
        merged = used_df.sort_values("monthnumerical").copy()

        if merged.empty:
            raise ValueError(f"No data found for ingredient: {selected_ingredient}.")

        # --- 4. Plot ---
        plt.figure(figsize=(12, 6))
        
        # Ensure plot columns are in the desired order
        merged_plot = merged[["month", "Used", "Shipped"]]
        
        plt.plot(merged_plot["month"], merged_plot["Used"], marker="o", label="Used")
        plt.plot(merged_plot["month"], merged_plot["Shipped"], marker="s", label="Shipped")
        
        plt.title(f"Used vs Shipped Over Time: {selected_ingredient}")
        plt.xlabel("Month")
        plt.ylabel("Amount (lbs or units)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close("all")

        # --- 5. Table Data ---
        
        # Rename 'Shipped' column and format numbers
        table_data = merged[["month", "Used", "Shipped"]].copy()
        table_data["Used"] = table_data["Used"].round(2)
        table_data["Shipped"] = table_data["Shipped"].round(2)
        
        table_records = table_data.to_dict("records")

        return jsonify({
            "image": img_b64,
            "note": f"Used vs Shipped timeline generated for {selected_ingredient}.",
            "table_data": table_records
        })

    except Exception as e:
        # Log full traceback to the console for debugging
        print("ERROR in /used_shipped_timeline_plot:", traceback.format_exc())

        # Return clean JSON error message
        return jsonify({
            "error": f"An error occurred: {str(e)}"
        }), 500


@app.route("/bestsellers_plot", methods=["POST"])
def bestsellers_plot():
    global item

    INGREDIENT_CONVERSION_MAP = {
        "braisedbeefusedg": "beef", 
        "braisedchickeng": "chicken", 
        "eggcount": "egg", 
        "riceg": "rice", 
        "ramencount": "ramen", 
        "ricenoodlesg": "rice noodles", 
        "chickenthighpcs": "chicken", 
        "chickenwingspcs": "chicken wings", 
        "flourg": "flour", 
        "greenonion": "green onion", 
        "cilantro": "cilantro", 
        "whiteonion": "white onion", 
        "peasg": "peas and carrot", 
        "bokchoyg": "bokchoy", 
        "tapiocastarch": "tapioca starch"
    }
    
    if item is None or item.empty:
        return jsonify({"error": "Item data not loaded."}), 400

    req = request.json or {}
    selected_month = req.get("month")

    # --- 1. Data Preparation and Filtering (omitted for brevity) ---
    # ... (code to filter df_filtered, df_top_10, and top_item_names remains the same) ...

    df_temp = item.copy()
    col_name_map = {col: normalize_text(col) for col in item.columns}
    df_temp.columns = df_temp.columns.map(col_name_map)
    
    if selected_month:
        month_col = 'month'
        month_filter = df_temp[month_col].astype(str).str.strip() == selected_month.strip()
        df_filtered = df_temp[month_filter].copy()
    else:
        df_filtered = df_temp.copy()

    df_filtered['amount'] = pd.to_numeric(df_filtered['amount'], errors='coerce').fillna(0)
    df_grouped = df_filtered.groupby('itemname')['amount'].sum().reset_index()
    df_top_10 = df_grouped.sort_values(by='amount', ascending=False).head(10).copy()
    
    if df_top_10.empty:
        return jsonify({"error": "No data available to determine top sellers."}), 400
    
    top_item_names = df_top_10['itemname'].tolist()
    df_ingredients = df_filtered[df_filtered['itemname'].isin(top_item_names)].copy()
    
    # --- 3. Ingredient Analysis for Top 10 Items ---
    
    ingredient_data = []
    all_bestseller_ingredients = [] # <-- NEW LIST TO TRACK ALL INGREDIENTS

    for item_name in top_item_names:
        df_item = df_ingredients[df_ingredients['itemname'] == item_name].copy()
        item_ingredients = []

        for ingredient_col in INGREDIENT_CONVERSION_MAP.keys():
            if ingredient_col in df_item.columns:
                total_usage = pd.to_numeric(df_item[ingredient_col], errors='coerce').fillna(0).sum()
                
                if total_usage > 0:
                    display_name = INGREDIENT_CONVERSION_MAP[ingredient_col].capitalize()
                    item_ingredients.append(display_name)
                    all_bestseller_ingredients.append(display_name) # <-- ADD TO FREQUENCY LIST
        
        if item_ingredients:
            ingredient_data.append({
                'Food Item': item_name,
                'Ingredients': item_ingredients
            })

    # --- 4. Ingredient Frequency Calculation (NEW) ---
    
    # Use Counter to get frequencies
    ingredient_counts = Counter(all_bestseller_ingredients)
    
    # Convert to list of dictionaries for JSON output
    # Sort by frequency descending
    frequency_table_data = [
        {'Ingredient': ing, 'Frequency': count} 
        for ing, count in ingredient_counts.most_common()
    ]

    # --- 5. Plot Generation (omitted for brevity) ---
    
    plt.figure(figsize=(12, 6))
    order_list = df_top_10['itemname'].tolist()
    sns.barplot(data=df_top_10, x='itemname', y='amount', order=order_list, palette='viridis')
    plt.title(f"Top 10 Bestselling Food Items (Filter: {selected_month if selected_month else 'All Months'})", fontsize=14)
    plt.xlabel("Food Item", fontsize=12)
    plt.ylabel("Total Amount ($)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Encoding plot to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close("all")

    # --- 6. Final Response ---
    
    return jsonify({
        "image": img_b64,
        "note": f"Top 10 bestsellers generated for {selected_month if selected_month else 'All Months'}.",
        "ingredient_table": ingredient_data,
        "frequency_table": frequency_table_data # <-- NEW DATA FIELD
    })
    

if __name__ == "__main__":
    app.run(debug=True, port = 5001)