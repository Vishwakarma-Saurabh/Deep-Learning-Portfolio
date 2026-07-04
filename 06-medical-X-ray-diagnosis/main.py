import os
import torch
from config import Config
from data import build_vocab, get_dataloaders
from models import ChestXRayClassifier, ReportGenerator
from training import train_classifier, train_report_generator, evaluate_classifier
from utils import plot_training_history, visualize_predictions

def main():
    """Main entry point"""
    Config.set_seed()
    
    # Create directories
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("🏥 MEDICAL CHEST X-RAY DIAGNOSIS SYSTEM")
    print("=" * 60)
    
    # Build vocabulary from sample reports
    sample_reports = [
        "Normal chest X-ray with no acute findings",
        "Cardiomegaly is present with pulmonary congestion",
        "Right lower lobe pneumonia with consolidation",
        "Pleural effusion noted at the right lung base",
        "Multiple findings: cardiomegaly, pulmonary edema"
    ]
    
    vocab, idx_to_word = build_vocab(sample_reports, max_size=Config.MAX_VOCAB_SIZE)
    
    # Create data loaders
    train_loader, test_loader = get_dataloaders(Config, vocab)
    
    # ============ CLASSIFIER ============
    print("\n🧠 Training Disease Classifier...")
    classifier = ChestXRayClassifier(
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.CLASSIFIER_PRETRAINED
    )
    
    classifier_history = train_classifier(
        classifier,
        train_loader,
        test_loader,
        Config
    )
    
    # Evaluate classifier
    evaluate_classifier(classifier, test_loader, Config.DEVICE)
    
    # Visualize predictions
    visualize_predictions(classifier, test_loader, Config.DEVICE)
    
    # Plot history
    plot_training_history(
        classifier_history,
        "Disease Classifier",
        f"{Config.OUTPUT_DIR}/classifier_history.png"
    )
    
    # ============ REPORT GENERATOR ============
    print("\n📝 Training Report Generator...")
    report_generator = ReportGenerator(
        vocab_size=len(vocab),
        embed_dim=Config.REPORT_EMBED_DIM,
        hidden_dim=Config.REPORT_HIDDEN_DIM
    )
    
    report_history = train_report_generator(
        report_generator,
        train_loader,
        test_loader,
        vocab,
        Config
    )
    
    # Plot history
    plot_training_history(
        report_history,
        "Report Generator",
        f"{Config.OUTPUT_DIR}/report_history.png"
    )
    
    # Generate sample reports
    print("\n📝 Generating Sample Reports...")
    report_generator.eval()
    with torch.no_grad():
        sample_images = next(iter(test_loader))[0][:4].to(Config.DEVICE)
        reports = report_generator.generate_report(
            sample_images,
            vocab,
            idx_to_word,
            max_length=50
        )
        print("\n📋 Generated Reports:")
        for i, report in enumerate(reports):
            print(f"  Report {i+1}: {report}")
    
    print("\n" + "=" * 60)
    print("🎉 PROJECT 6 COMPLETED!")
    print("=" * 60)
    print("🏥 System can now:")
    print("  1. Detect 14 chest diseases from X-rays")
    print("  2. Generate clinical reports")
    print("=" * 60)


if __name__ == "__main__":
    main()