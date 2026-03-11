from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .services import fetch_stock_data, prepare_data_for_regression
from .models_logic import run_linear_regression, run_logistic_regression, run_arima_model, run_lstm_model

class StockPredictionView(APIView):
    def post(self, request):
        symbol = request.data.get('symbol')
        model_type = request.data.get('model_type') # 'regression', 'arima', 'lstm'
        
        if not symbol:
            return Response({"error": "Symbol is required"}, status=status.HTTP_400_BAD_REQUEST)
            
        df, err = fetch_stock_data(symbol)
        if err:
            return Response({"error": err}, status=status.HTTP_400_BAD_REQUEST)
            
        # Get last 30 days of history for the graph
        history = df.tail(30)[['Date', 'Close']].to_dict('records')
        # Convert Date objects to strings for JSON serialization
        for h in history:
            h['Date'] = h['Date'].strftime('%Y-%m-%d') if hasattr(h['Date'], 'strftime') else str(h['Date'])
            
        if model_type == 'regression':
            df_reg = prepare_data_for_regression(df)
            lin_preds, lin_metrics = run_linear_regression(df_reg)
            log_preds, log_metrics = run_logistic_regression(df_reg)
            return Response({
                "history": history,
                "linear": {"predictions": lin_preds, "metrics": lin_metrics},
                "logistic": {"predictions": log_preds, "metrics": log_metrics}
            })
        
        elif model_type == 'arima':
            preds, metrics = run_arima_model(df)
            return Response({"history": history, "predictions": preds, "metrics": metrics})
            
        elif model_type == 'lstm':
            preds, metrics = run_lstm_model(df)
            return Response({"history": history, "predictions": preds, "metrics": metrics})
            
        return Response({"error": "Invalid model type"}, status=status.HTTP_400_BAD_REQUEST)
