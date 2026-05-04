from langchain_openai import ChatOpenAI
from pathlib import Path
import pandas as pd
import os
import json
from typing import Dict, List, Tuple
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import re
from collections import Counter

load_dotenv()

class BatchReportGenerator:
    def __init__(self, batch_size: int = 50, max_workers: int = 3, delay_between_batches: float = 1.0):
        """
        Initialize batch processor for clinical report generation.
        
        Args:
            batch_size: Number of patients to process per batch
            max_workers: Number of concurrent threads for LLM calls
            delay_between_batches: Delay in seconds between batches
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.delay_between_batches = delay_between_batches
        self.variation_styles_used = []  # To track which styles were used

        # Define diverse clinical note templates focusing on findings and impressions only
        self.suspicious_templates = {
            'intro': [
                "Irregular enhancement patterns observed in {side} breast.",
                "Complex enhancement kinetics identified in the {side} breast.",
                "The {side} breast demonstrates a lesion with asymmetric enhancement patterns.",
                "Focal area of concern in {side} breast showing suspicious enhancement characteristics.",
                "Enhancement abnormality detected in {side} breast with concerning kinetic curve patterns.",
                "Areas of non-mass enhancement identified in {side} breast.",
                "The {side} breast shows clustered enhancement.",
                "Asymmetric enhancement detected in {side} breast region."
            ],
            'finding': [
                "Heterogeneous morphological characteristics noted with rapid contrast uptake.",
                "Irregular margins and architectural distortion were noted.",
                "Spiculated margins and architectural distortion are present.",
                "Irregular morphology was observed.",
                "Segmental distribution of the enhancement is visible.",
                "Irregular spatial distribution and kinetic characteristics are concerning.",
                "Associated architectural changes are visible."
            ]
        }

        self.normal_templates = {
            'intro': [
                "Normal parenchymal enhancement patterns in {side} breast.",
                "The {side} breast demonstrates physiological enhancement.",
                "Imaging of the {side} breast reveals normal enhancement kinetics.",
                "Assessment of the {side} breast shows normal morphological patterns.",
                "The findings in {side} breast are within normal limits.",
                "Symmetric fibroglandular tissue distribution observed in {side} breast.",
                "The {side} breast exhibits normal parenchymal characteristics.",
                "Enhancement patterns in {side} breast demonstrate typical morphological features."
            ],
            'finding': [
                "Symmetric fibroglandular tissue distribution observed.",
                "Homogeneous parenchymal pattern with no focal lesions identified.",
                "Scattered fibroglandular tissue with background enhancement within normal limits.",
                "Symmetric enhancement with no suspicious lesions detected.",
                "Typical parenchymal enhancement with no architectural distortion or focal abnormalities noted.",
                "Physiological enhancement patterns are seen.",
                "Homogeneous background enhancement is noted.",
                "Symmetric distribution and no suspicious characteristics identified."
            ]
        }

    def load_and_process_data(self, table_path: Path) -> pd.DataFrame:
        """Load and process the split CSV data."""
        try:
            table = pd.read_csv(table_path)
            print(f"Loaded {len(table)} records from {table_path}")
            return table
        except FileNotFoundError:
            print(f"File not found: {table_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def analyze_imaging_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze imaging patterns across the dataset."""
        if df.empty:
            return {}
        
        analysis = {}

        # Extract patient ID and side from UID
        df['patient_id'] = df['UID'].str.extract(r'(\d+)_')
        df['side'] = df['UID'].str.extract(r'_(\w+)$')
        
        # Group by patient ID to analyze bilateral cases
        for patient_id in df['patient_id'].unique():
            patient_data = df[df['patient_id'] == patient_id]
            
            # Analyze each side
            sides_analysis = {}
            for _, row in patient_data.iterrows():
                uid = row['UID']
                malignant = row['Malignant']
                fold = row['Fold']
                split = row['Split']
                side = row['side']
                
                # Determine findings category without diagnostic terms
                findings_category = "SUSPICIOUS_FINDINGS" if malignant == 1 else "NORMAL_FINDINGS"
                
                # Generate clinical interpretation focused on observations
                if malignant == 1:
                    intro = random.choice(self.suspicious_templates['intro'])
                    finding = random.choice(self.suspicious_templates['finding'])
                    clinical_note = f"{intro.format(side=side)} {finding}"
                else:
                    intro = random.choice(self.normal_templates['intro'])
                    finding = random.choice(self.normal_templates['finding'])
                    clinical_note = f"{intro.format(side=side)} {finding}"
                
                sides_analysis[uid] = {
                    'side': side,
                    'suspicious_indicator': malignant,
                    'findings_category': findings_category,
                    'fold': fold,
                    'split': split,
                    'clinical_note': clinical_note
                }
            
            analysis[patient_id] = sides_analysis
        
        return analysis

    def create_patient_batches(self, analysis: Dict) -> List[List[str]]:
        """Create batches of patient IDs for processing."""
        patient_ids = list(analysis.keys())
        random.shuffle(patient_ids)  # Randomize order for better load balancing
        
        batches = []
        for i in range(0, len(patient_ids), self.batch_size):
            batch = patient_ids[i:i + self.batch_size]
            batches.append(batch)
        
        print(f"Created {len(batches)} batches of up to {self.batch_size} patients each")
        return batches

    def generate_detailed_report(self, patient_id: str, patient_data: Dict, llm: ChatOpenAI, batch_num: int) -> Tuple[str, str, str]:
        """Generate detailed clinical report for a patient using LLM."""
        
        # Prepare context for LLM
        context = f"Patient ID: {patient_id}\n"
        context += f"Number of studies: {len(patient_data)}\n"

        bilateral_findings = []
        for uid, data in patient_data.items():
            context += f"\nUID: {uid}"
            context += f"\nSide: {data['side']}"
            context += f"\nFindings Category: {data['findings_category']}"
            context += f"\nFold: {data['fold']}, Split: {data['split']}"
            context += f"\nClinical Observations: {data['clinical_note']}\n"
            bilateral_findings.append(data['findings_category'])
        
        # Determine bilateral presentation
        if len(set(bilateral_findings)) > 1:
            bilateral_note = "ASYMMETRIC PRESENTATION - Unilateral area of concern identified"
        elif "SUSPICIOUS_FINDINGS" in bilateral_findings:
            bilateral_note = "BILATERAL AREAS OF CONCERN"
        else:
            bilateral_note = "BILATERAL NORMAL FINDINGS - No suspicious lesions identified"

        # Add diverse instructions for the LLM with batch-specific variations
        synthesis_styles = [
            "Synthesize the imaging findings into a descriptive summary focusing on morphological characteristics and enhancement patterns, followed by objective observations.",
            "Provide a narrative-style report describing the imaging features and enhancement kinetics without diagnostic interpretations.",
            "Structure the report with clear 'Imaging Findings' and 'Technical Observations' sections, focusing on what is visualized rather than diagnostic conclusions.",
            "Generate a report emphasizing morphological descriptions and enhancement characteristics. Focus on objective findings and imaging features.",
            "Create a comprehensive imaging overview describing enhancement patterns, tissue characteristics, and morphological features for each breast.",
            "Develop a systematic assessment highlighting spatial distribution of enhancement and morphological patterns observed.",
            "Formulate a technical report focusing on kinetic characteristics and enhancement morphology without diagnostic language.",
            "Construct a detailed findings summary emphasizing imaging characteristics and enhancement distribution patterns."
        ]
        
        # Use batch number to add variation
        synthesis_instruction = synthesis_styles[batch_num % len(synthesis_styles)]
        
        prompt = f"""
        Generate a comprehensive clinical imaging report for a breast MRI case for patient {patient_id}.
        
        **IMPORTANT GUIDELINES:**
        - Focus ONLY on imaging findings, morphological descriptions, and enhancement patterns
        - Do NOT include diagnostic terms like "malignant," "benign," "cancer," "tumor," or "normal"
        - Use descriptive terms like "area of concern," "suspicious findings," "enhancement abnormality," "within expected limits"
        - Describe what is observed without making diagnostic conclusions
        - Use objective, descriptive language throughout
        
        **Patient Imaging Context:**
        {context}
        
        **Bilateral Assessment:**
        {bilateral_note}
        
        **Report Structure Instructions:**
        {synthesis_instruction}
        
        Please structure your response with the following sections:
        1. Imaging Summary
        2. Morphological Findings
        3. Enhancement Characteristics
        
        Maintain a professional, descriptive tone focused on imaging observations rather than diagnostic interpretations.
        """
        
        try:
            response = llm.invoke(prompt)
            # Return patient_id, report content, and the style used
            return patient_id, response.content, synthesis_instruction
        except Exception as e:
            error_report = f"Error generating detailed report: {e}\n\nBasic Findings Report:\n{context}\nBilateral Findings: {bilateral_note}"
            return patient_id, error_report, "Error"

    def process_batch(self, batch_patient_ids: List[str], analysis: Dict, llm: ChatOpenAI, batch_num: int) -> Dict[str, str]:
        """Process a batch of patients concurrently."""
        batch_reports = {}
        used_in_batch = set() # Track generated notes to ensure uniqueness
        
        print(f"Processing batch {batch_num + 1} with {len(batch_patient_ids)} patients...")
        
        # Regenerate notes for this batch to ensure uniqueness
        for patient_id in batch_patient_ids:
            for uid, data in analysis[patient_id].items():
                malignant = data['suspicious_indicator']
                side = data['side']
                
                # Generate unique clinical note
                while True:
                    if malignant == 1:
                        templates = self.suspicious_templates
                        intro = random.choice(templates['intro'])
                        finding = random.choice(templates['finding'])
                        note_tuple = (intro, finding)
                        if note_tuple not in used_in_batch:
                            used_in_batch.add(note_tuple)
                            data['clinical_note'] = f"{intro.format(side=side)} {finding}"
                            break
                    else:
                        templates = self.normal_templates
                        intro = random.choice(templates['intro'])
                        finding = random.choice(templates['finding'])
                        note_tuple = (intro, finding)
                        if note_tuple not in used_in_batch:
                            used_in_batch.add(note_tuple)
                            data['clinical_note'] = f"{intro.format(side=side)} {finding}"
                            break

        # Process patients in this batch concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_patient = {
                executor.submit(self.generate_detailed_report, patient_id, analysis[patient_id], llm, batch_num): patient_id
                for patient_id in batch_patient_ids
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_patient):
                patient_id, report, style_used = future.result()
                batch_reports[patient_id] = report
                if style_used != "Error":
                    self.variation_styles_used.append(style_used)
                completed += 1
                if completed % 10 == 0:
                    print(f"  Completed {completed}/{len(batch_patient_ids)} in batch {batch_num + 1}")
        
        print(f"Batch {batch_num + 1} completed successfully")
        return batch_reports

    def generate_reports_in_batches(self, analysis: Dict, llm: ChatOpenAI, output_dir: Path) -> Dict[str, str]:
        """Generate detailed reports for all patients in batches."""
        if not llm:
            print("No LLM available for detailed report generation")
            return {}
        
        # Create batches
        batches = self.create_patient_batches(analysis)
        all_detailed_reports = {}
        
        # Process each batch
        for batch_num, batch_patient_ids in enumerate(batches):
            try:
                # Process batch
                batch_reports = self.process_batch(batch_patient_ids, analysis, llm, batch_num)
                all_detailed_reports.update(batch_reports)
                
                # Save intermediate results
                batch_output_path = output_dir / f'batch_{batch_num + 1}_reports.json'
                with open(batch_output_path, 'w') as f:
                    json.dump(batch_reports, f, indent=2)
                print(f"Batch {batch_num + 1} results saved to: {batch_output_path}")
                
                # Delay between batches to avoid rate limits
                if batch_num < len(batches) - 1:  # Don't delay after the last batch
                    print(f"Waiting {self.delay_between_batches} seconds before next batch...")
                    time.sleep(self.delay_between_batches)
                    
            except Exception as e:
                print(f"Error processing batch {batch_num + 1}: {e}")
                continue
        
        print(f"Completed processing all {len(batches)} batches. Total reports generated: {len(all_detailed_reports)}")
        return all_detailed_reports

    def analyze_report_variations(self, detailed_reports: Dict) -> Dict:
        """Analyze the generated reports for length and vocabulary diversity."""
        if not detailed_reports:
            return {}
            
        report_lengths = [len(report) for report in detailed_reports.values()]
        total_words = []
        for report in detailed_reports.values():
            words = re.findall(r'\w+', report.lower())
            total_words.extend(words)
        
        unique_words = set(total_words)
        total_word_count = len(total_words)
        
        diversity_ratio = len(unique_words) / total_word_count if total_word_count > 0 else 0
        
        return {
            "avg_length": sum(report_lengths) / len(report_lengths) if report_lengths else 0,
            "vocabulary_diversity": {
                "unique_words": len(unique_words),
                "total_words": total_word_count,
                "diversity_ratio": diversity_ratio
            }
        }
    
    def save_reports(self, analysis: Dict, output_dir: Path, llm: ChatOpenAI = None):
        """Save individual and summary reports with batch processing."""
        output_dir.mkdir(exist_ok=True)
        
        # Generate detailed reports in batches
        detailed_reports = {}
        if llm:
            detailed_reports = self.generate_reports_in_batches(analysis, llm, output_dir)
        
        # Generate summary table
        summary_data = []
        for patient_id, patient_data in analysis.items():
            # Add to summary
            for uid, data in patient_data.items():
                summary_data.append({
                    'Patient_ID': patient_id,
                    'UID': uid,
                    'Side': data['side'],
                    'Suspicious_Indicator': data['suspicious_indicator'],
                    'Findings_Category': data['findings_category'],
                    'Fold': data['fold'],
                    'Split': data['split'],
                    'Clinical_Observations': data['clinical_note']
                })
        
        # Save summary table
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / 'imaging_findings_summary_table.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary table saved to: {summary_path}")
        
        # Save consolidated detailed reports with variation analysis
        if detailed_reports:
            reports_path = output_dir / 'detailed_imaging_reports_consolidated.json'
            
            # Add variation analysis to the output
            variation_analysis_data = self.analyze_report_variations(detailed_reports)
            
            output_data = {
                "reports": detailed_reports,
                "variation_analysis": variation_analysis_data,
                "processing_metadata": {
                    "batch_size": self.batch_size,
                    "max_workers": self.max_workers,
                    "total_batches_processed": len(detailed_reports) // self.batch_size + (1 if len(detailed_reports) % self.batch_size else 0),
                    "styles_used": list(Counter(self.variation_styles_used).items())
                }
            }
            
            with open(reports_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Consolidated detailed reports with variation analysis saved to: {reports_path}")
        
        # Generate enhanced markdown report
        markdown_path = output_dir / 'imaging_findings_analysis_report.md'
        self.generate_markdown_report(summary_df, detailed_reports, markdown_path)
        print(f"Enhanced markdown report saved to: {markdown_path}")

    def generate_markdown_report(self, summary_df: pd.DataFrame, detailed_reports: Dict, output_path: Path):
        """Generate a comprehensive markdown report with variation analysis."""
        
        with open(output_path, 'w') as f:
            f.write("# Breast MRI Imaging Findings Analysis Report\n\n")
            f.write(f"*Generated using enhanced batch processing with intelligent variation*\n\n")
            
            # Dataset statistics
            f.write("## Dataset Statistics\n\n")
            f.write(f"- **Total Cases**: {len(summary_df['Patient_ID'].unique())} patients\n")
            f.write(f"- **Total UIDs**: {len(summary_df)} bilateral assessments\n")
            f.write(f"- **Cases with Suspicious Findings**: {len(summary_df[summary_df['Suspicious_Indicator'] == 1])}\n")
            f.write(f"- **Cases with Normal Findings**: {len(summary_df[summary_df['Suspicious_Indicator'] == 0])}\n")
            f.write(f"- **Detailed Reports Generated**: {len(detailed_reports)}\n\n")
            
            # Processing statistics with variation info
            f.write("## Processing & Variation Statistics\n\n")
            f.write(f"- **Batch Size**: {self.batch_size} patients per batch\n")
            f.write(f"- **Max Concurrent Workers**: {self.max_workers}\n")
            f.write(f"- **Delay Between Batches**: {self.delay_between_batches} seconds\n")
            
            if detailed_reports:
                variation_analysis = self.analyze_report_variations(detailed_reports)
                f.write(f"- **Average Report Length**: {variation_analysis.get('avg_length', 0):.0f} characters\n")
                f.write(f"- **Vocabulary Diversity**: {variation_analysis.get('vocabulary_diversity', {}).get('diversity_ratio', 0):.3f}\n")
                f.write(f"- **Style Variations Used**: {len(set(self.variation_styles_used))}\n")
            
            f.write("\n")
            
            # Style distribution
            if self.variation_styles_used:
                f.write("## Report Style Distribution\n\n")
                style_counts = Counter(self.variation_styles_used)
                
                for style, count in sorted(style_counts.items()):
                    base_style = style.split('.', 1)[-1] if '.' in style else style
                    f.write(f"- **{base_style.replace('_', ' ')}**: {count} reports\n")
                f.write("\n")
            
            # Findings distribution
            f.write("## Findings Distribution by Side\n\n")
            side_stats = summary_df.groupby(['Side', 'Findings_Category']).size().unstack(fill_value=0)
            f.write(side_stats.to_markdown())
            f.write("\n\n")
            
            # Summary table (first 20 cases)
            f.write("## Summary Table (Sample)\n\n")
            f.write(summary_df.head(20).to_markdown(index=False))
            f.write("\n\n")
            
            # Detailed reports with style annotations
            if detailed_reports:
                f.write("## Detailed Imaging Reports (Sample with Style Variations)\n\n")
                
                # Pair reports with their styles
                reports_with_styles = list(zip(detailed_reports.items(), self.variation_styles_used))
                
                for i, ((patient_id, report), style_used) in enumerate(reports_with_styles[:5]):
                    style_name = style_used.replace('_', ' ')
                    f.write(f"### Patient {patient_id} (*Style: {style_name}*)\n\n")
                    f.write(report)
                    f.write("\n\n---\n\n")

def main():
    """Main execution function with batch processing."""
    # Configuration
    table_path = Path(r'\\rad-maid-004\D\Duke-Cancer_MRI\preprocessed_crop-a\splits\split.csv')
    output_dir = Path('./imaging_findings_reports')
    
    # Batch processing configuration
    BATCH_SIZE = 25  # Adjust based on your needs and rate limits
    MAX_WORKERS = 2  # Adjust based on your API rate limits
    DELAY_BETWEEN_BATCHES = 2.0  # Seconds between batches
    
    # Initialize batch processor
    batch_processor = BatchReportGenerator(
        batch_size=BATCH_SIZE,
        max_workers=MAX_WORKERS,
        delay_between_batches=DELAY_BETWEEN_BATCHES
    )
    
    # Initialize LLM (optional - requires API key)
    llm = None
    try:
        if os.getenv('OPENAI_API_KEY'):
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
            print("LLM initialized for detailed report generation")
        else:
            print("No OpenAI API key found. Generating basic reports only.")
    except Exception as e:
        print(f"Could not initialize LLM: {e}")
    
    # Load and process data
    print("Loading data...")
    df = batch_processor.load_and_process_data(table_path)
    
    if df.empty:
        print("No data loaded. Exiting.")
        return
    
    # Analyze imaging patterns
    print("Analyzing imaging patterns...")
    analysis = batch_processor.analyze_imaging_patterns(df)
    
    if not analysis:
        print("No analysis results. Exiting.")
        return
    
    print(f"Total patients to process: {len(analysis)}")
    
    # Generate and save reports with batch processing
    print("Generating reports in batches...")
    batch_processor.save_reports(analysis, output_dir, llm)
    
    print(f"Batch report generation completed! Check {output_dir} for results.")


if __name__ == "__main__":
    main()