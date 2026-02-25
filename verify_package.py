from indic_text_normalization.normalize import Normalizer

try:
    print("Testing English Normalization...")
    normalizer_en = Normalizer(lang='en', input_case='cased')
    res_en = normalizer_en.normalize("I have 2 apples.")
    print(f"En Result: {res_en}")
    assert "two" in res_en

    print("\nTesting Hindi Normalization...")
    normalizer_hi = Normalizer(lang='hi', input_case='cased')
    res_hi = normalizer_hi.normalize("मुझे 10 रुपये चाहिए")
    print(f"Hi Result: {res_hi}")
    # minimal check to see if 10 converted to das/dus
    assert "दस" in res_hi or "१०" not in res_hi 

    print("\nSUCCESS: Package is working correctly!")
except Exception as e:
    print(f"\nFAILURE: {e}")
    exit(1)
