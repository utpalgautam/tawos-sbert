import torch
import torch.nn as nn


class StoryPointCNN(nn.Module):
    """
    1D CNN regression model for predicting software story points
    from frozen SBERT embeddings.

    Input:
        Tensor of shape (batch_size, embedding_dim)

    Output:
        Tensor of shape (batch_size,), containing continuous predictions.
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        conv_channels: tuple = (32, 64, 128),
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Store configuration
        self.embedding_dim = embedding_dim
        self.conv_channels = conv_channels

        # Validate configuration
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        if not conv_channels:
            raise ValueError("conv_channels cannot be empty.")

        if any(
            not isinstance(channels, int) or channels <= 0
            for channels in conv_channels
        ):
            raise ValueError(
                "Every value in conv_channels must be a positive integer."
            )

        if not isinstance(kernel_size, int) or kernel_size <= 0:
            raise ValueError("kernel_size must be a positive integer.")

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        # CNN feature extractor
        cnn_layers = []
        in_channels = 1

        for out_channels in conv_channels:
            cnn_layers.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                ]
            )

            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # Convert the sequence of CNN features into one feature vector
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)

        # Regression head
        final_channels = conv_channels[-1]

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_channels, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Run a batch of SBERT embeddings through the CNN.

        Args:
            embeddings:
                Float tensor of shape (batch_size, embedding_dim).

        Returns:
            Float tensor of shape (batch_size,).
        """

        if embeddings.ndim != 2:
            raise ValueError(
                "Expected a 2D tensor with shape "
                f"(batch_size, {self.embedding_dim}); "
                f"received {tuple(embeddings.shape)}."
            )

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"but received {embeddings.shape[1]}."
            )

        # (batch_size, embedding_dim)
        #          -> (batch_size, 1, embedding_dim)
        x = embeddings.unsqueeze(1)

        # Extract local features
        x = self.cnn(x)

        # Pool over the embedding sequence dimension
        x = self.global_pool(x)

        # Predict one continuous story-point value per issue
        predictions = self.regressor(x)

        # (batch_size, 1) -> (batch_size,)
        return predictions.squeeze(-1)


if __name__ == "__main__":
    # Basic architecture smoke test
    torch.manual_seed(42)

    model = StoryPointCNN()

    dummy_embeddings = torch.randn(16, 384)
    predictions = model(dummy_embeddings)

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print("Model:", model.__class__.__name__)
    print("Input shape:", tuple(dummy_embeddings.shape))
    print("Output shape:", tuple(predictions.shape))
    print("Trainable parameters:", parameter_count)
    print("Sample predictions:", predictions[:5].detach().tolist())

    assert predictions.shape == (16,)
    assert torch.isfinite(predictions).all()

    print("Model test passed!")