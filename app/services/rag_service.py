import os
from dotenv import load_dotenv
import google.generativeai as genai
import requests
import json

class RAGService:
    def __init__(self):
        load_dotenv()

        # ---- Gemini Setup ----
        gemini_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        genai.configure(api_key=gemini_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        self.gemini_vision_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Backend URL for scheme search
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:5000")

    # ---------------- Gemini functions ----------------
    def search_schemes(self, query, language="English", user_profile=None):
        """Enhanced RAG with vectorized scheme search"""
        # Get vectorized schemes from backend
        relevant_schemes = self._get_vectorized_schemes(query)
        
        # Enhanced government services context
        gov_context = self._get_government_services_context()
        
        prompt = f"""
        You are an Indian Government Services Assistant. Answer directly without preambles or disclaimers.
        
        User Query: {query}
        Language: {language}
        
        Relevant Schemes Found:
        {relevant_schemes}
        
        Government Services Context:
        {gov_context}
        
        Instructions:
        - Answer the query directly
        - List relevant schemes with eligibility and benefits
        - Include required documents and application process
        - No introductory text or disclaimers
        - Be concise and helpful
        - Respond in {language}
        """
        
        response = self.gemini_model.generate_content(prompt)
        return response.text

    def generate_form_help(self, fields, language="English"):
        """Enhanced form filling assistance"""
        prompt = f"""
        You are a government form filling assistant for India.
        
        Form Fields: {fields}
        Language: {language}
        
        Provide step-by-step guidance including:
        1. What information is needed for each field
        2. Where to find required documents
        3. Common mistakes to avoid
        4. Tips for faster processing
        
        Be helpful and explain in simple {language}.
        """
        response = self.gemini_model.generate_content(prompt)
        return response.text
    
    def generate_comprehensive_form_help(self, extracted_text, detected_fields, document_type, language="English"):
        """Generate comprehensive form filling assistance using OCR results"""
        
        # Format fields for better context
        fields_info = []
        for field in detected_fields:
            fields_info.append(f"- {field.get('field', 'Unknown')}: {field.get('type', 'text')} field")
        
        fields_text = "\n".join(fields_info) if fields_info else "No specific fields detected"
        
        prompt = f"""
        You are an expert Indian government form filling assistant. Help the user fill out this form based on the OCR analysis.
        
        DOCUMENT ANALYSIS:
        Document Type: {document_type}
        Extracted Text: {extracted_text[:500]}...
        
        DETECTED FORM FIELDS:
        {fields_text}
        
        TASK: Provide comprehensive form filling guidance in {language}
        
        Include:
        1. **Document Identification**: What type of form this appears to be
        2. **Required Information**: What details are needed for each field
        3. **Document Requirements**: Which supporting documents to prepare
        4. **Step-by-Step Instructions**: How to fill each section
        5. **Common Mistakes**: What errors to avoid
        6. **Processing Tips**: How to ensure faster approval
        
        Make it practical and actionable. Use simple {language}.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"RAG service error: {e}")
            # Try simpler fallback
            try:
                return self._generate_simple_form_help(detected_fields, document_type, language)
            except:
                # Final fallback
                return f"I can help you fill this form. Based on the analysis, this appears to be a {document_type} document. Please ensure you have all required documents ready and fill the fields carefully."
    
    def analyze_form_image_directly(self, image_data, language="English"):
        """Analyze form image directly using Gemini Vision"""
        try:
            import base64
            from PIL import Image
            import io
            
            # Handle base64 image data
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                header, base64_data = image_data.split(',', 1)
                image_bytes = base64.b64decode(base64_data)
            elif isinstance(image_data, str):
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data
            
            # Convert to PIL Image and ensure RGB format
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            prompt = f"""
Analyze this government form image and provide comprehensive form filling guidance in {language}.

Please identify:
1. What type of form this is
2. What fields need to be filled
3. What documents are required
4. Step-by-step filling instructions
5. Common mistakes to avoid

Be practical and helpful. Respond in {language}.
"""
            
            # Use vision model with proper content format
            response = self.gemini_vision_model.generate_content([prompt, image])
            return response.text
            
        except Exception as e:
            print(f"Direct image analysis failed: {e}")
            # Try alternative approach
            try:
                return self._analyze_with_fallback(image_data, language)
            except:
                return None
    
    def _generate_simple_form_help(self, fields, doc_type, language="English"):
        """Simple fallback form help without AI"""
        help_parts = []
        help_parts.append(f"Form Type: {doc_type.replace('_', ' ').title()}")
        help_parts.append("\nRequired Information:")
        
        for field in fields:
            field_type = field.get('type', 'text')
            if field_type == 'name':
                help_parts.append("- Full Name: Write your complete name as per official documents")
            elif field_type == 'email':
                help_parts.append("- Email: Provide a valid email address")
            elif field_type == 'phone':
                help_parts.append("- Phone: 10-digit mobile number")
            elif field_type == 'address':
                help_parts.append("- Address: Complete postal address with PIN code")
            elif field_type == 'date':
                help_parts.append("- Date: Use DD/MM/YYYY format")
            else:
                help_parts.append(f"- {field.get('field', 'Field')}: Fill accurately")
        
        help_parts.append("\nGeneral Tips:")
        help_parts.append("- Use black/blue pen only")
        help_parts.append("- Write clearly in capital letters")
        help_parts.append("- Do not leave mandatory fields blank")
        help_parts.append("- Attach required documents")
        
        return "\n".join(help_parts)
    
    def _analyze_with_fallback(self, image_data, language="English"):
        """Fallback image analysis method"""
        try:
            import base64
            
            # Simple text-based analysis
            prompt = f"""
I need help analyzing a government form image for form filling guidance in {language}.

Please provide:
1. General form filling tips
2. Common document requirements
3. Step-by-step guidance
4. Mistakes to avoid

Respond in {language}.
"""
            
            response = self.gemini_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Fallback analysis failed: {e}")
            return "Unable to analyze the form image. Please ensure the image is clear and try again."
            
    def get_universal_help(self, query, language="English"):
        """Universal government services helper"""
        return self.search_schemes(query, language)

    # ---------------- Helpers ----------------
    def _get_vectorized_schemes(self, query):
        """Fetch relevant schemes using vector search"""
        try:
            response = requests.post(
                f"{self.backend_url}/api/v1/schemes/search",
                json={"query": query},
                timeout=10
            )
            if response.status_code == 200:
                schemes = response.json().get("schemes", [])
                return self._format_schemes_for_context(schemes)
        except Exception as e:
            print(f"Vector search failed: {e}")
        
        # Fallback to basic schemes data
        return self._get_basic_schemes_data()
    
    def _format_schemes_for_context(self, schemes):
        """Format schemes data for AI context"""
        if not schemes:
            return "No specific schemes found for this query."
        
        formatted = []
        for scheme in schemes[:5]:  # Limit to top 5
            formatted.append(f"""
            Scheme: {scheme.get('name', 'N/A')}
            Overview: {scheme.get('overview', 'N/A')[:200]}...
            Eligibility: {scheme.get('eligibility', 'N/A')[:150]}...
            Benefits: {scheme.get('benefits', 'N/A')[:150]}...
            Documents: {scheme.get('documents', 'N/A')[:100]}...
            """)
        
        return "\n".join(formatted)
    
    def _get_government_services_context(self):
        """Universal government services knowledge base"""
        return """
        INSURANCE SERVICES:
        - Pradhan Mantri Jeevan Jyoti Bima Yojana (Life Insurance - ₹2 lakh)
        - Pradhan Mantri Suraksha Bima Yojana (Accident Insurance - ₹2 lakh)
        - Pradhan Mantri Fasal Bima Yojana (Crop Insurance)
        - Ayushman Bharat (Health Insurance - ₹5 lakh)
        
        HEALTHCARE SERVICES:
        - AIIMS hospitals and government medical colleges
        - Primary Health Centers (PHCs) and Community Health Centers
        - Jan Aushadhi stores for affordable medicines
        - National Health Mission programs
        
        EDUCATION & SCHOLARSHIPS:
        - National Scholarship Portal (scholarships.gov.in)
        - PM YASASVI Scheme for OBC/EBC/DNT students
        - Post Matric Scholarship for SC/ST/OBC
        - Merit-cum-Means Scholarship
        
        EMPLOYMENT & SKILLS:
        - MGNREGA (100 days guaranteed employment)
        - Pradhan Mantri Kaushal Vikas Yojana (Skill Development)
        - Startup India and Stand Up India
        - Rozgar Mela (Government job fairs)
        
        DIGITAL SERVICES:
        - Aadhaar services and updates
        - PAN card application and services
        - Passport services (passportindia.gov.in)
        - Driving license and vehicle registration
        - Income/caste/domicile certificates
        
        FINANCIAL SERVICES:
        - Jan Dhan Yojana (Bank accounts)
        - PM Mudra Yojana (Business loans)
        - Kisan Credit Card
        - Direct Benefit Transfer (DBT)
        
        SOCIAL WELFARE:
        - Public Distribution System (PDS/Ration)
        - Widow pension schemes
        - Disability pension and certificates
        - Senior citizen benefits
        """
    
    def _get_basic_schemes_data(self):
        """Fallback schemes data when vector search fails"""
        return """
        PM-KISAN: ₹6,000/year for farmers, Land records + Aadhaar required
        Ayushman Bharat: ₹5 lakh health insurance for BPL families
        PM Mudra Yojana: Business loans up to ₹10 lakh
        MGNREGA: 100 days guaranteed employment in rural areas
        PM Awas Yojana: Housing assistance for eligible families
        
        FORM FILLING GUIDANCE:
        - Aadhaar Card: 12-digit unique identification number
        - PAN Card: 10-character alphanumeric code for tax purposes
        - Passport: For international travel documentation
        - Driving License: For vehicle operation authorization
        - Voter ID: For electoral participation
        - Birth Certificate: Proof of birth and age
        
        COMMON DOCUMENTS NEEDED:
        - Address proof (utility bills, rent agreement)
        - Identity proof (Aadhaar, PAN, passport)
        - Income proof (salary slips, ITR)
        - Photographs (passport size)
        - Bank account details
        
        TIPS FOR FORM FILLING:
        - Use black or blue pen only
        - Write in capital letters clearly
        - Do not leave mandatory fields blank
        - Attach all required documents
        - Keep photocopies of all documents
        - Verify all information before submission
        """