import os
import re
import pandas as pd
import asyncio
from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')


model = ChatOpenAI(model_name="gpt-4o", temperature=0.0, api_key=os.getenv('OPENAI_API_KEY'))
prompt = """
Extract the laterality (left, right, or bilateral) of breast lesions/tumors from the following radiology narrative report.

TASK:
Analyze the provided radiology report and identify all mentions of breast lesions, masses, tumors, or other localized findings (e.g., biopsy clips, artifacts, architectural distortion). For each finding, determine its laterality based on explicit anatomical references.

LATERALITY CLASSIFICATION CODES:
0 = UNSPECIFIED: Lesions mentioned without clear laterality indication
1 = RIGHT: Lesions specifically described as being in the right breast
2 = LEFT: Lesions specifically described as being in the left breast  
3 = BILATERAL: Lesions present in both breasts

KEY INDICATORS TO LOOK FOR:

Right Breast Indicators (Code: 1):
- "right breast"
- "right breast artifact"
- "right axillary" 
- "right upper/lower/outer/inner quadrant"
- "right retroareolar"
- "right subareolar"
- Anatomical descriptions following "Right:" section headers

Left Breast Indicators (Code: 2):
- "left breast"
- "left breast artifact"
- "left axillary"
- "left upper/lower/outer/inner quadrant"
- "left retroareolar"
- "left subareolar"
- Anatomical descriptions following "Left:" section headers

Bilateral Indicators (Code: 3):
- "bilateral" or "bilaterally" when describing a finding, not the procedure.
- "both breasts" when describing findings.
- Lesions explicitly described in both the left and right breast sections.

ANALYSIS GUIDELINES:

Context Clues:
- Pay attention to section headers (e.g., "Left:", "Right:")
- Consider anatomical quadrant descriptions
- Note axillary lymph node involvement
- Look for comparative language ("compared to the contralateral breast")

Common Pitfalls to Avoid:
- Don't consider the e
- Don't assume laterality from previous mentions without explicit confirmation
- Be careful with pronouns - ensure they refer to the correct anatomical side
- Watch for lesions described in relation to biopsy clips or markers
- Consider that multiple lesions may have different lateralities
- **Distinguish Procedure vs. Finding:** The imaging technique (e.g., 'Bilateral MRI') may be bilateral, but this does not mean the findings are bilateral. Base your laterality code ONLY on the location of the actual lesions, masses, or suspicious findings described in the report.
- **Include All Findings:** Do not ignore a finding just because it is described as 'benign,' 'stable,' or is a post-procedural change (e.g., 'biopsy artifact'). If a finding has a specific location, its laterality must be coded.

Ambiguous Cases:
- If laterality cannot be definitively determined, classify as UNSPECIFIED (Code: 0)
- Note any contextual information that might suggest laterality
- Distinguish between primary tumors and metastatic involvement

Now analyze the following radiology report and return ONLY the single-digit LATERALITY CODE.

Radiology Report:
{radiology_report}
"""



def define_laterality():
    
    df = pd.read_csv(r"D:\Users\UFPB\gabriel ayres\MST\table_utils\dummy_ehr_new.csv")
    df['Narrative'] = df['Narrative'].astype(str)
    dummy_df = df[df['laterality'].isna()].copy()
    cont = 0
    for index, row in dummy_df.iterrows():
        if cont == 10:
            break
        if pd.notna(row['laterality']):
            continue
        try:
            laterality_output = model.invoke(prompt.format(radiology_report=row['Narrative'])).content
            print(row['Narrative'])
            print(f"LLM Output: {laterality_output}")
            # Find the first digit in the output
            match = re.search(r'\d+', laterality_output)
            if match:
                laterality_code = match.group(0)
                dummy_df.loc[index, 'laterality'] = float(laterality_code)
                print(f"Found code: {laterality_code}")
            else:
                dummy_df.loc[index, 'laterality'] = 'Parse Error'
                print("Could not find laterality code.")

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            dummy_df.loc[index, 'laterality'] = 'Error'
        cont += 1

    dummy_df.to_csv(r"D:\Users\UFPB\gabriel ayres\MST\table_utils\test_dummy_ehr_new.csv", index=False)

    comparison_df = dummy_df[['Narrative', 'laterality']]
    comparison_df.to_csv(r"D:\Users\UFPB\gabriel ayres\MST\table_utils\comparison_dummy_ehr.csv", index=False)
    print("Done")

if __name__ == "__main__":
    define_laterality()
