#!/usr/bin/env python3
"""
Training log analysis tool for debugging and parameter tuning assistance.
Generates concise reports from training analysis logs for Claude to review.
"""

import os
import json
import sys
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd


class TrainingLogAnalyzer:
    """
    Analyzes training logs and generates debugging reports.
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize the log analyzer.
        
        Args:
            log_dir: Directory containing training logs
        """
        self.log_dir = Path(log_dir)
        self.analysis_dir = self.log_dir / "analysis"
        self.algorithm = "unknown"
        self.analysis_files = []
        self.config = {}
        
        # Load configuration and find analysis files
        self._load_config()
        self._find_analysis_files()
        
    def _load_config(self) -> None:
        """Load training configuration."""
        config_path = self.analysis_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                self.algorithm = self.config.get('algorithm', 'unknown')
        else:
            # Try to detect algorithm from directory name
            if "ppo" in str(self.log_dir).lower():
                self.algorithm = "ppo"
            elif "dapg" in str(self.log_dir).lower():
                self.algorithm = "dapg"
                
    def _find_analysis_files(self) -> None:
        """Find all analysis files in the log directory."""
        if not self.analysis_dir.exists():
            print(f"Warning: Analysis directory not found: {self.analysis_dir}")
            return
            
        pattern = str(self.analysis_dir / "analysis_*.json")
        self.analysis_files = sorted(glob.glob(pattern))
        
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive analysis report.
        
        Args:
            output_file: Optional file to save the report
            
        Returns:
            Report string
        """
        if not self.analysis_files:
            return "No analysis files found. Make sure training was run with analysis callback."
        
        # Load all analysis data
        analyses = []
        for file_path in self.analysis_files:
            try:
                with open(file_path, 'r') as f:
                    analyses.append(json.load(f))
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        if not analyses:
            return "No valid analysis files found."
        
        # Generate report sections
        report_sections = [
            self._generate_header(analyses),
            self._generate_training_summary(analyses),
            self._generate_performance_analysis(analyses),
            self._generate_health_analysis(analyses),
            self._generate_recommendations_summary(analyses),
            self._generate_trend_analysis(analyses),
            self._generate_debugging_insights(analyses),
            self._generate_conclusion(analyses)
        ]
        
        report = "\n".join(report_sections)
        
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_file}")
        
        return report
    
    def _generate_header(self, analyses: List[Dict]) -> str:
        """Generate report header."""
        first_analysis = analyses[0]
        last_analysis = analyses[-1]
        
        header = f"""
# Training Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Training Session Overview
- **Algorithm**: {self.algorithm.upper()}
- **Log Directory**: {self.log_dir}
- **Training Duration**: {last_analysis['elapsed_time']:.1f} seconds
- **Total Timesteps**: {last_analysis['timestep']:,}
- **Analysis Points**: {len(analyses)}
- **Analysis Interval**: {self.config.get('log_interval', 'unknown')} steps
"""
        
        if self.config.get('model_config'):
            header += f"\n## Model Configuration\n"
            for key, value in self.config['model_config'].items():
                header += f"- **{key}**: {value}\n"
        
        return header
    
    def _generate_training_summary(self, analyses: List[Dict]) -> str:
        """Generate training summary."""
        latest = analyses[-1]
        metrics = latest.get('metrics', {})
        
        summary = f"""
## Training Summary

### Latest Performance (Step {latest['timestep']:,})
"""
        
        # Episode reward summary
        if 'episode_reward' in metrics:
            reward_info = metrics['episode_reward']
            summary += f"- **Episode Reward**: {reward_info['mean']:.3f} ± {reward_info['std']:.3f}\n"
            summary += f"  - Range: [{reward_info['min']:.3f}, {reward_info['max']:.3f}]\n"
            summary += f"  - Episodes: {reward_info['count']}\n"
        
        # Success rate
        if 'success_rate' in metrics:
            success_info = metrics['success_rate']
            summary += f"- **Success Rate**: {success_info['rate']:.3f} ({success_info['count']} episodes)\n"
        
        # Training losses
        if 'policy_loss' in metrics:
            policy_info = metrics['policy_loss']
            summary += f"- **Policy Loss**: {policy_info['mean']:.6f} (trend: {policy_info['trend']})\n"
        
        if 'value_loss' in metrics:
            value_info = metrics['value_loss']
            summary += f"- **Value Loss**: {value_info['mean']:.6f} (trend: {value_info['trend']})\n"
        
        # DAPG-specific metrics
        if 'bc_loss' in metrics:
            bc_info = metrics['bc_loss']
            summary += f"- **BC Loss**: {bc_info['mean']:.6f} (trend: {bc_info['trend']})\n"
        
        return summary
    
    def _generate_performance_analysis(self, analyses: List[Dict]) -> str:
        """Generate performance analysis."""
        # Extract performance metrics over time
        timesteps = []
        rewards = []
        success_rates = []
        
        for analysis in analyses:
            timesteps.append(analysis['timestep'])
            metrics = analysis.get('metrics', {})
            
            if 'episode_reward' in metrics:
                rewards.append(metrics['episode_reward']['mean'])
            
            if 'success_rate' in metrics:
                success_rates.append(metrics['success_rate']['rate'])
        
        analysis_text = f"""
## Performance Analysis

### Reward Progression
"""
        
        if rewards:
            analysis_text += f"- **Initial Reward**: {rewards[0]:.3f}\n"
            analysis_text += f"- **Final Reward**: {rewards[-1]:.3f}\n"
            analysis_text += f"- **Improvement**: {rewards[-1] - rewards[0]:.3f}\n"
            
            # Simple trend analysis
            if len(rewards) > 5:
                recent_trend = np.polyfit(range(len(rewards[-5:])), rewards[-5:], 1)[0]
                if recent_trend > 0.01:
                    analysis_text += f"- **Recent Trend**: Improving (+{recent_trend:.3f}/analysis)\n"
                elif recent_trend < -0.01:
                    analysis_text += f"- **Recent Trend**: Declining ({recent_trend:.3f}/analysis)\n"
                else:
                    analysis_text += f"- **Recent Trend**: Stable\n"
        
        if success_rates:
            analysis_text += f"\n### Success Rate Progression\n"
            analysis_text += f"- **Initial Success Rate**: {success_rates[0]:.3f}\n"
            analysis_text += f"- **Final Success Rate**: {success_rates[-1]:.3f}\n"
            analysis_text += f"- **Improvement**: {success_rates[-1] - success_rates[0]:.3f}\n"
        
        return analysis_text
    
    def _generate_health_analysis(self, analyses: List[Dict]) -> str:
        """Generate training health analysis."""
        health_analysis = f"""
## Training Health Analysis

### Health Status Over Time
"""
        
        # Count health issues
        total_issues = 0
        issue_types = {}
        critical_issues = []
        
        for analysis in analyses:
            health = analysis.get('health_check', {})
            if health.get('issues'):
                total_issues += len(health['issues'])
                for issue in health['issues']:
                    issue_type = issue['type']
                    issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
                    
                    if issue['severity'] == 'critical':
                        critical_issues.append({
                            'step': analysis['timestep'],
                            'issue': issue
                        })
        
        if total_issues == 0:
            health_analysis += "- ✅ **No health issues detected**\n"
        else:
            health_analysis += f"- ⚠️ **Total Issues**: {total_issues}\n"
            health_analysis += f"- **Issue Types**:\n"
            for issue_type, count in issue_types.items():
                health_analysis += f"  - {issue_type}: {count} occurrences\n"
        
        if critical_issues:
            health_analysis += f"\n### Critical Issues\n"
            for critical in critical_issues:
                health_analysis += f"- **Step {critical['step']:,}**: {critical['issue']['message']}\n"
        
        return health_analysis
    
    def _generate_recommendations_summary(self, analyses: List[Dict]) -> str:
        """Generate recommendations summary."""
        recommendations_summary = f"""
## Recommendations Summary

### Most Recent Recommendations
"""
        
        # Get latest recommendations
        latest_recommendations = analyses[-1].get('recommendations', [])
        
        if not latest_recommendations:
            recommendations_summary += "- No specific recommendations at this time\n"
        else:
            # Group by type and priority
            high_priority = [r for r in latest_recommendations if r.get('priority') == 'high']
            medium_priority = [r for r in latest_recommendations if r.get('priority') == 'medium']
            
            if high_priority:
                recommendations_summary += f"#### High Priority\n"
                for rec in high_priority:
                    recommendations_summary += f"- **{rec['type']}**: {rec['reason']}\n"
            
            if medium_priority:
                recommendations_summary += f"#### Medium Priority\n"
                for rec in medium_priority:
                    recommendations_summary += f"- **{rec['type']}**: {rec['reason']}\n"
        
        # Analyze recommendation trends
        rec_counts = {}
        for analysis in analyses:
            for rec in analysis.get('recommendations', []):
                rec_type = rec['type']
                rec_counts[rec_type] = rec_counts.get(rec_type, 0) + 1
        
        if rec_counts:
            recommendations_summary += f"\n### Recurring Recommendations\n"
            for rec_type, count in sorted(rec_counts.items(), key=lambda x: x[1], reverse=True):
                recommendations_summary += f"- **{rec_type}**: {count} times\n"
        
        return recommendations_summary
    
    def _generate_trend_analysis(self, analyses: List[Dict]) -> str:
        """Generate trend analysis."""
        trend_analysis = f"""
## Trend Analysis

### Key Metrics Trends
"""
        
        # Analyze policy loss trend
        policy_losses = []
        for analysis in analyses:
            metrics = analysis.get('metrics', {})
            if 'policy_loss' in metrics:
                policy_losses.append(metrics['policy_loss']['mean'])
        
        if policy_losses:
            trend_analysis += f"#### Policy Loss\n"
            if len(policy_losses) > 3:
                trend_slope = np.polyfit(range(len(policy_losses)), policy_losses, 1)[0]
                if trend_slope > 0.001:
                    trend_analysis += f"- 📈 **Increasing trend** (slope: {trend_slope:.6f})\n"
                elif trend_slope < -0.001:
                    trend_analysis += f"- 📉 **Decreasing trend** (slope: {trend_slope:.6f})\n"
                else:
                    trend_analysis += f"- ➡️ **Stable trend**\n"
            
            trend_analysis += f"- Initial: {policy_losses[0]:.6f}, Final: {policy_losses[-1]:.6f}\n"
        
        # Analyze BC loss trend for DAPG
        if self.algorithm == "dapg":
            bc_losses = []
            for analysis in analyses:
                metrics = analysis.get('metrics', {})
                if 'bc_loss' in metrics:
                    bc_losses.append(metrics['bc_loss']['mean'])
            
            if bc_losses:
                trend_analysis += f"#### BC Loss\n"
                if len(bc_losses) > 3:
                    trend_slope = np.polyfit(range(len(bc_losses)), bc_losses, 1)[0]
                    if trend_slope > 0.001:
                        trend_analysis += f"- 📈 **Increasing trend** (slope: {trend_slope:.6f})\n"
                    elif trend_slope < -0.001:
                        trend_analysis += f"- 📉 **Decreasing trend** (slope: {trend_slope:.6f})\n"
                    else:
                        trend_analysis += f"- ➡️ **Stable trend**\n"
                
                trend_analysis += f"- Initial: {bc_losses[0]:.6f}, Final: {bc_losses[-1]:.6f}\n"
        
        return trend_analysis
    
    def _generate_debugging_insights(self, analyses: List[Dict]) -> str:
        """Generate debugging insights."""
        insights = f"""
## Debugging Insights

### Parameter Tuning Suggestions
"""
        
        # Analyze common patterns
        latest_metrics = analyses[-1].get('metrics', {})
        
        # Learning rate analysis
        if 'policy_loss' in latest_metrics:
            policy_trend = latest_metrics['policy_loss']['trend']
            if policy_trend == 'increasing':
                insights += f"- 🔧 **Learning Rate**: Consider reducing (policy loss increasing)\n"
            elif policy_trend == 'stable' and latest_metrics['policy_loss']['mean'] > 1.0:
                insights += f"- 🔧 **Learning Rate**: Consider increasing (slow convergence)\n"
        
        # Reward analysis
        if 'episode_reward' in latest_metrics:
            reward_std = latest_metrics['episode_reward']['std']
            if reward_std > 100:
                insights += f"- 🔧 **Batch Size**: Consider increasing (high reward variance: {reward_std:.3f})\n"
        
        # Algorithm-specific insights
        if self.algorithm == "dapg" and 'bc_loss' in latest_metrics:
            bc_loss_mean = latest_metrics['bc_loss']['mean']
            if bc_loss_mean > 2.0:
                insights += f"- 🔧 **BC Weight**: Consider increasing (BC loss high: {bc_loss_mean:.6f})\n"
            elif bc_loss_mean < 0.1:
                insights += f"- 🔧 **BC Weight**: Consider decreasing (BC loss very low: {bc_loss_mean:.6f})\n"
        
        # Training stability analysis
        warnings_count = sum(len(a.get('warnings', [])) for a in analyses)
        if warnings_count > len(analyses) * 0.5:
            insights += f"- ⚠️ **Training Stability**: Many warnings detected ({warnings_count}), consider reducing learning rate\n"
        
        return insights
    
    def _generate_conclusion(self, analyses: List[Dict]) -> str:
        """Generate conclusion and next steps."""
        conclusion = f"""
## Conclusion & Next Steps

### Training Assessment
"""
        
        latest_metrics = analyses[-1].get('metrics', {})
        
        # Overall assessment
        if 'episode_reward' in latest_metrics:
            reward_mean = latest_metrics['episode_reward']['mean']
            if reward_mean > 0:
                conclusion += f"- ✅ **Overall**: Training showing positive rewards ({reward_mean:.3f})\n"
            else:
                conclusion += f"- ⚠️ **Overall**: Training struggling with negative rewards ({reward_mean:.3f})\n"
        
        if 'success_rate' in latest_metrics:
            success_rate = latest_metrics['success_rate']['rate']
            if success_rate > 0.8:
                conclusion += f"- ✅ **Success Rate**: Excellent performance ({success_rate:.3f})\n"
            elif success_rate > 0.5:
                conclusion += f"- ✅ **Success Rate**: Good performance ({success_rate:.3f})\n"
            else:
                conclusion += f"- ⚠️ **Success Rate**: Needs improvement ({success_rate:.3f})\n"
        
        # Next steps
        conclusion += f"""
### Recommended Next Steps
1. **Continue Training**: Monitor progress for convergence
2. **Parameter Adjustment**: Review recommendations above
3. **Validation**: Test current best checkpoint
4. **Analysis**: Re-run this analysis after parameter changes

### Files to Review
- Latest checkpoint: `{self.log_dir}/checkpoints/`
- TensorBoard logs: `{self.log_dir}/tensorboard_logs/`
- Analysis logs: `{self.analysis_dir}/`
"""
        
        return conclusion
    
    def generate_quick_summary(self) -> str:
        """Generate a quick summary for immediate review."""
        if not self.analysis_files:
            return "No analysis data available."
        
        # Load latest analysis
        with open(self.analysis_files[-1], 'r') as f:
            latest = json.load(f)
        
        summary = f"""
QUICK TRAINING SUMMARY ({self.algorithm.upper()})
Step: {latest['timestep']:,} | Time: {latest['elapsed_time']:.1f}s
"""
        
        metrics = latest.get('metrics', {})
        if 'episode_reward' in metrics:
            reward = metrics['episode_reward']['mean']
            summary += f"Reward: {reward:.3f} | "
        
        if 'success_rate' in metrics:
            success = metrics['success_rate']['rate']
            summary += f"Success: {success:.3f} | "
        
        if 'policy_loss' in metrics:
            policy_loss = metrics['policy_loss']['mean']
            summary += f"Policy Loss: {policy_loss:.6f}"
        
        # Health status
        health = latest.get('health_check', {})
        if health.get('status') != 'healthy':
            summary += f"\n⚠️ Health: {health['status']}"
        
        # Top recommendation
        recommendations = latest.get('recommendations', [])
        if recommendations:
            top_rec = recommendations[0]
            summary += f"\n💡 Suggestion: {top_rec['reason']}"
        
        return summary


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Analyze training logs for debugging")
    parser.add_argument("log_dir", help="Directory containing training logs")
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument("--quick", action="store_true", help="Generate quick summary only")
    
    args = parser.parse_args()
    
    # Check if log directory exists
    if not os.path.exists(args.log_dir):
        print(f"Error: Log directory not found: {args.log_dir}")
        sys.exit(1)
    
    # Create analyzer
    analyzer = TrainingLogAnalyzer(args.log_dir)
    
    # Generate report
    if args.quick:
        report = analyzer.generate_quick_summary()
    else:
        report = analyzer.generate_report(args.output)
    
    print(report)


if __name__ == "__main__":
    main()