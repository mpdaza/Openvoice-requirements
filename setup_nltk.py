import nltk
nltk.data.path.append("/content/py39/nltk_data")
nltk.download("punkt", download_dir="/content/py39/nltk_data")
nltk.download("averaged_perceptron_tagger", download_dir="/content/py39/nltk_data")
nltk.download("averaged_perceptron_tagger_eng", download_dir="/content/py39/nltk_data")
print("✅ NLTK descargado correctamente")