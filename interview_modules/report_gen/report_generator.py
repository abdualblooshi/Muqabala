from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from datetime import datetime
from typing import List, Dict
import arabic_reshaper
from bidi.algorithm import get_display
import os

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
    def generate_report(self,
                       candidate_name: str,
                       position: str,
                       interview_date: datetime,
                       questions_and_answers: List[Dict],
                       evaluation_metrics: Dict,
                       output_path: str,
                       language: str = 'en') -> str:
        """Generate a PDF report of the interview."""
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build the story (content)
        story = []
        
        # Add title
        title = "Interview Evaluation Report"
        if language == 'ar':
            title = "تقرير تقييم المقابلة"
            title = get_display(arabic_reshaper.reshape(title))
        story.append(Paragraph(title, self.title_style))
        
        # Add candidate information
        story.extend(self._add_candidate_info(
            candidate_name,
            position,
            interview_date,
            language
        ))
        
        # Add questions and answers
        story.extend(self._add_qa_section(
            questions_and_answers,
            language
        ))
        
        # Add evaluation metrics
        story.extend(self._add_evaluation_metrics(
            evaluation_metrics,
            language
        ))
        
        # Build the PDF
        doc.build(story)
        return output_path
    
    def _add_candidate_info(self,
                          name: str,
                          position: str,
                          date: datetime,
                          language: str) -> List:
        """Add candidate information section."""
        content = []
        
        if language == 'ar':
            name = get_display(arabic_reshaper.reshape(name))
            position = get_display(arabic_reshaper.reshape(position))
            
        info_table_data = [
            ['Candidate Name:', name],
            ['Position:', position],
            ['Interview Date:', date.strftime('%Y-%m-%d %H:%M')]
        ]
        
        if language == 'ar':
            info_table_data = [
                ['اسم المرشح:', name],
                ['المنصب:', position],
                ['تاريخ المقابلة:', date.strftime('%Y-%m-%d %H:%M')]
            ]
        
        table = Table(info_table_data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 0), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(table)
        content.append(Spacer(1, 20))
        return content
    
    def _add_qa_section(self,
                       qa_list: List[Dict],
                       language: str) -> List:
        """Add questions and answers section."""
        content = []
        
        # Add section title
        title = "Interview Questions and Responses"
        if language == 'ar':
            title = "أسئلة وأجوبة المقابلة"
            title = get_display(arabic_reshaper.reshape(title))
        content.append(Paragraph(title, self.styles['Heading2']))
        content.append(Spacer(1, 12))
        
        # Add each Q&A pair
        for i, qa in enumerate(qa_list, 1):
            q_prefix = f"Q{i}: " if language == 'en' else f"س{i}: "
            a_prefix = "A: " if language == 'en' else "ج: "
            
            question = qa['question']
            answer = qa['answer']
            
            if language == 'ar':
                question = get_display(arabic_reshaper.reshape(question))
                answer = get_display(arabic_reshaper.reshape(answer))
            
            content.append(Paragraph(
                f"{q_prefix}{question}",
                self.styles['Heading3']
            ))
            content.append(Paragraph(
                f"{a_prefix}{answer}",
                self.styles['Normal']
            ))
            content.append(Spacer(1, 12))
        
        return content
    
    def _add_evaluation_metrics(self,
                              metrics: Dict,
                              language: str) -> List:
        """Add evaluation metrics section."""
        content = []
        
        # Add section title
        title = "Evaluation Metrics"
        if language == 'ar':
            title = "مقاييس التقييم"
            title = get_display(arabic_reshaper.reshape(title))
        content.append(Paragraph(title, self.styles['Heading2']))
        content.append(Spacer(1, 12))
        
        # Create metrics table
        metrics_data = [
            ['Metric', 'Score'],
            ['Relevance', f"{metrics['relevance_score']:.2f}"],
            ['Confidence', f"{metrics['confidence_score']:.2f}"],
            ['Technical', f"{metrics['technical_score']:.2f}"],
            ['Language', f"{metrics['language_score']:.2f}"],
            ['Overall', f"{metrics['overall_score']:.2f}"]
        ]
        
        if language == 'ar':
            metrics_data = [
                ['المقياس', 'النتيجة'],
                ['الصلة', f"{metrics['relevance_score']:.2f}"],
                ['الثقة', f"{metrics['confidence_score']:.2f}"],
                ['التقني', f"{metrics['technical_score']:.2f}"],
                ['اللغة', f"{metrics['language_score']:.2f}"],
                ['الإجمالي', f"{metrics['overall_score']:.2f}"]
            ]
            metrics_data = [[get_display(arabic_reshaper.reshape(str(cell))) 
                           if isinstance(cell, str) else cell 
                           for cell in row] 
                          for row in metrics_data]
        
        table = Table(metrics_data, colWidths=[3*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER')
        ]))
        
        content.append(table)
        content.append(Spacer(1, 20))
        
        # Add feedback
        feedback_title = "Feedback"
        feedback = metrics.get('feedback', '')
        if language == 'ar':
            feedback_title = "التعليقات"
            feedback = get_display(arabic_reshaper.reshape(feedback))
            feedback_title = get_display(arabic_reshaper.reshape(feedback_title))
        
        content.append(Paragraph(feedback_title, self.styles['Heading3']))
        content.append(Paragraph(feedback, self.styles['Normal']))
        
        return content 