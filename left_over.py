def find_temporal_universal_circuit(trained_lstm, dataloader, window_size, num_layers, num_experts, device, lambda_reg=0.05, epochs=50):
    """
    Optimizes a mask over an aligned temporal window (e.g., the last N tokens of a prompt)
    to find the sequence-specific safety circuit.
    """
    trained_lstm.eval()
    for param in trained_lstm.parameters():
        param.requires_grad = False

    # 1. Initialize a mask that explicitly models the temporal window
    # Shape: (1, window_size, num_layers, num_experts)
    windowed_mask = nn.Parameter(torch.ones(1, window_size, num_layers, num_experts, device=device))

    optimizer = optim.Adam([windowed_mask], lr=0.05)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0

        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # batch_x MUST be pre-aligned and truncated/padded to exactly `window_size`
            # Shape: (batch_size, window_size, num_layers, num_experts)

            constrained_mask = torch.clamp(windowed_mask, 0.0, 1.0)

            # Mask applies to specific temporal positions relative to the end of the prompt
            masked_input = batch_x * constrained_mask

            # Flatten spatial dimensions for LSTM
            batch_size = masked_input.size(0)
            lstm_input = masked_input.view(batch_size, window_size, -1)

            logits = trained_lstm(lstm_input)

            cls_loss = criterion(logits, batch_y)
            l1_loss = lambda_reg * torch.sum(torch.abs(constrained_mask))

            loss = cls_loss + l1_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            active_nodes = (windowed_mask > 0.5).sum().item()
            print(f"Epoch {epoch} | Loss: {total_loss/len(dataloader):.4f} | Active Circuit Nodes: {active_nodes}")

    final_mask = torch.clamp(windowed_mask, 0.0, 1.0).detach().squeeze()

    # Returns shape (window_size, num_layers, num_experts)
    temporal_circuit = (final_mask > 0.5).int()

    return temporal_circuit, final_mask