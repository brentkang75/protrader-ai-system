"""
Startup fix untuk memastikan semua dependency ter-load dengan benar
"""
import sys
import os

# Tambahkan current directory ke Python path
sys.path.append(os.path.dirname(__file__))

def preload_modules():
    """Preload modules yang sering error"""
    try:
        import pandas, numpy, sklearn, xgboost, ta
        print("✅ Semua modules utama berhasil di-load")
        return True
    except ImportError as e:
        print(f"❌ Module error: {e}")
        return False

if __name__ == "__main__":
    preload_modules()
