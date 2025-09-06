#!/usr/bin/env python3
"""
Test script to debug form analysis issues
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.ocr_service import OCRService
from services.rag_service import RAGService
from services.translation_service import TranslationService
from services.tts_service import TTSService

def test_form_analysis():
    print("🧪 Testing Form Analysis Pipeline...")
    
    # Test OCR Service
    print("\n1. Testing OCR Service...")
    try:
        ocr_service = OCRService()
        print("✅ OCR Service initialized")
        
        # Test with a simple base64 image (1x1 white pixel)
        test_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        ocr_result = ocr_service.extract_text_from_image(test_image)
        print(f"OCR Result: {ocr_result}")
        
    except Exception as e:
        print(f"❌ OCR Service failed: {e}")
        return False
    
    # Test RAG Service
    print("\n2. Testing RAG Service...")
    try:
        rag_service = RAGService()
        print("✅ RAG Service initialized")
        
        # Test comprehensive form help
        help_text = rag_service.generate_comprehensive_form_help(
            extracted_text="Name: _____ Age: _____ Address: _____",
            detected_fields=[{"field": "Name", "type": "name"}, {"field": "Age", "type": "number"}],
            document_type="application_form",
            language="English"
        )
        print(f"Form Help Generated: {help_text[:100]}...")
        
    except Exception as e:
        print(f"❌ RAG Service failed: {e}")
        return False
    
    # Test Translation Service
    print("\n3. Testing Translation Service...")
    try:
        translation_service = TranslationService()
        print("✅ Translation Service initialized")
        
    except Exception as e:
        print(f"❌ Translation Service failed: {e}")
        return False
    
    # Test TTS Service
    print("\n4. Testing TTS Service...")
    try:
        tts_service = TTSService()
        print("✅ TTS Service initialized")
        
    except Exception as e:
        print(f"❌ TTS Service failed: {e}")
        return False
    
    print("\n🎉 All services initialized successfully!")
    return True

if __name__ == "__main__":
    test_form_analysis()