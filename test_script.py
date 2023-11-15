import torch
from transformers import PerceiverModel, PerceiverConfig
class CrossAttentionPerceiver(torch.nn.Module):
    def __init__(self, config):
        super(CrossAttentionPerceiver, self).__init__()

        self.perceiver = PerceiverModel(config)
        
    def forward(self, input1, input2, attention_mask1=None, attention_mask2=None):
        """
        Args:
            input1: Input tensor for the first sequence.
            input2: Input tensor for the second sequence.
            attention_mask1: Attention mask for the first sequence (optional).
            attention_mask2: Attention mask for the second sequence (optional).
        Returns:
            outputs: Model outputs.
        """
        # Assuming both inputs have the same sequence length
        input_shape = input1.size()

        # Combine the two sequences
        combined_input = torch.cat([input1, input2], dim=1)

        # Combine the attention masks if provided
        if attention_mask1 is not None and attention_mask2 is not None:
            combined_attention_mask = torch.cat([attention_mask1, attention_mask2], dim=1)
        else:
            combined_attention_mask = None

        # Forward pass through the Perceiver model
        outputs = self.perceiver(
            inputs_embeds=combined_input,
            attention_mask=combined_attention_mask
        )

        return outputs

# Example usage:
config = PerceiverConfig(
        num_latents=256,
        d_latents=1280,
        d_model=768,
        num_blocks=1,
        num_self_attends_per_block=26,
        num_self_attention_heads=8,
        num_cross_attention_heads=8,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
        vocab_size=262,
        max_position_embeddings=2048,
        image_size=56,
        train_size=[368, 496],
        num_frames=16,
        audio_samples_per_frame=1920,
        samples_per_patch=16,
        output_shape=[1, 16, 224, 224],
        output_num_channels=512,
        _label_trainable_num_channels=1024)
cross_attention_perceiver =  PerceiverModel(config)

# Dummy input tensors
input1 = torch.randn(1, 10, 768)  # Sequence length of 10, assuming embedding size is 768
input2 = torch.randn(1, 10, 768)

# Forward pass
outputs = cross_attention_perceiver(torch.cat([input1, input2], dim=1))
print(outputs['last_hidden_state'].shape)
