# from flask import Flask
# from .api import approval_bp, test_bp
# from .middleware import setup_metrics, before_request, after_request
#
#
# def create_app():
#     """
#         Create and configure the Flask application instance.
#
#         Returns:
#             Flask: The configured Flask Application Instance.
#     """
#
#     app = Flask(__name__)
#
#     # Register the Blueprints with the app
#     app.register_blueprint(approval_bp)
#
#     # Register the Blueprints with the app
#     app.register_blueprint(test_bp)
#
#     # Register Prometheus metrics middleware
#     setup_metrics(app)
#
#     # Register before and after request handlers
#     app.before_request(before_request)
#     app.after_request(after_request)
#
#     return app