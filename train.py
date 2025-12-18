import torch
from torch import nn, optim
from tqdm import tqdm
import os
import json

def train_model(model, train_loader, val_loader,
                num_epochs=10, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu",
                val_dataset=None, replay_loader=None):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        replay_iter = iter(replay_loader) if replay_loader else None
        ### 🔁 TRAINING LOOP ###
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        replay_loss_total = 0.0
        replay_count = 0

        for batch_idx, (images, labels) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")):
            images, labels = images.to(device), labels.to(device)

            # 🔁 Normal batch update
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # 🧠 Replay every 5 steps
            if replay_iter and batch_idx % 5 == 0:
                try:
                    replay_images, replay_labels = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    replay_images, replay_labels = next(replay_iter)

                replay_images, replay_labels = replay_images.to(device), replay_labels.to(device)
                replay_outputs = model(replay_images)
                replay_loss = criterion(replay_outputs, replay_labels)

                optimizer.zero_grad()
                replay_loss.backward()
                optimizer.step()

                # Logging
                replay_loss_total += replay_loss.item() * replay_images.size(0)
                replay_count += replay_images.size(0)

                if batch_idx % 10 == 0:
                    print(f"[Replay] Step {batch_idx} | Replay loss: {replay_loss.item():.4f}")

        if replay_count > 0:
            avg_replay_loss = replay_loss_total / replay_count
            print(f"Replay Loss: {avg_replay_loss:.4f}")
        train_acc = correct / total
        train_loss /= total

        ### 🧪 VALIDATION LOOP WITH CONFIDENCE LOGGING ###
        if val_loader is not None and val_dataset is not None:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            predictions_log = []

            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    probs = torch.softmax(outputs, dim=1)
                    confs, preds = torch.max(probs, 1)

                    for i in range(images.size(0)):
                        sample_index = batch_idx * val_loader.batch_size + i
                        if sample_index >= len(val_dataset.samples):
                            continue
                        image_path, true_label_idx = val_dataset.samples[sample_index]
                        predictions_log.append({
                            "image_path": str(image_path),
                            "true_label": true_label_idx,
                            "predicted_label": preds[i].item(),
                            "confidence": float(confs[i].item()),
                            "correct": bool(true_label_idx == preds[i].item())
                        })

                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total
            val_loss /= total

            #print(f"\n📊 Epoch {epoch + 1} Summary:")
           # print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
            #print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc * 100:.2f}%\n")

        # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "checkpoints/best_model.pth")
                print("💾 Saved best model!\n")

                with open("checkpoints/best_val_predictions.json", "w") as f:
                    json.dump(predictions_log, f, indent=2)

