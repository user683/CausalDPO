import warnings
from collections import defaultdict
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, PreTrainedTokenizerBase
from sklearn.cluster import DBSCAN
from .utils import DPODataCollatorWithPadding, pad_to_length


def is_peft_available():
    return importlib.util.find_spec("peft") is not None


if is_peft_available():
    from peft import get_peft_model


class DPOTrainer(Trainer):
    def __init__(
            self,
            model,
            ref_model,
            beta=0.1,
            args=None,
            data_collator=None,
            tokenizer=None,
            train_dataset=None,
            eval_dataset=None,
            max_length=512,
            max_prompt_length=128,
            label_pad_token_id=-100,
            padding_value=0,
            peft_config=None,
            num_clusters=2,  # Desired number of clusters (for initialization) 2
            mmd_lambda=0.001,  # MMD regularization weight
            mmd_kernel="rbf",  # Kernel type
            mmd_sigma=1.0,  # Bandwidth of RBF kernel
            eps=0.5,  # Neighborhood radius for DBSCAN 0.5
            min_samples=10,  # Minimum samples for DBSCAN core points
    ):
        if peft_config and is_peft_available():
            model = get_peft_model(model, peft_config)
        elif peft_config:
            raise ValueError("PEFT is not available but peft_config was provided.")

        if data_collator is None:
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided if data_collator is None.")
            data_collator = DPODataCollatorWithPadding(
                tokenizer, max_length=max_length, max_prompt_length=max_prompt_length,
                label_pad_token_id=label_pad_token_id, padding_value=padding_value
            )
            if args.remove_unused_columns:
                args.remove_unused_columns = False
                warnings.warn("Set remove_unused_columns=False for DPODataCollatorWithPadding.", UserWarning)
            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value
        self.beta = beta
        self.ref_model = ref_model
        self.num_clusters = num_clusters
        self.mmd_lambda = mmd_lambda
        self.mmd_kernel = mmd_kernel
        self.mmd_sigma = mmd_sigma
        self.eps = eps
        self.min_samples = min_samples
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer)

        self.causal_feature_extractor = nn.Linear(model.config.hidden_size, model.config.hidden_size).to(self.accelerator.device)
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, model.config.hidden_size).to(self.accelerator.device))

        # reference model
        self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def extract_causal_features(self, batch):
        outputs = self.model(
            batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1][:, 0, :]  # Take [CLS] position (batch_size, hidden_size)
        causal_features = self.causal_feature_extractor(hidden_states)
        return causal_features

    def cluster_batch(self, batch):
        "Causal Soft Clustering: Probabilistic Modeling via Causal Features"
        causal_features = self.extract_causal_features(batch)  # (batch_size, hidden_size)

        with torch.no_grad():
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = dbscan.fit_predict(causal_features.cpu().numpy())
            hard_labels = torch.tensor(labels, device=self.accelerator.device)
            unique_labels = torch.unique(hard_labels[hard_labels >= 0])
            if len(unique_labels) > 0:
                initial_centers = torch.stack([causal_features[hard_labels == l].mean(dim=0) for l in unique_labels])
                if len(initial_centers) < self.num_clusters:
                    extra_centers = torch.randn(self.num_clusters - len(initial_centers), causal_features.shape[1]).to(
                        self.accelerator.device)
                    initial_centers = torch.cat([initial_centers, extra_centers], dim=0)
                self.cluster_centers.data[:len(initial_centers)] = initial_centers[:self.num_clusters]

        distances = torch.cdist(causal_features, self.cluster_centers)  # (batch_size, num_clusters)
        probs = F.softmax(-distances, dim=-1)  # (batch_size, num_clusters)
        return probs

    def group_by_env(self, batch, cluster_probs):
        env_groups = defaultdict(dict)
        batch_size = batch["chosen_input_ids"].shape[0]
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                for env_id in range(self.num_clusters):
                    weights = cluster_probs[:, env_id].view(batch_size, 1, *(1 for _ in v.shape[1:]))
                    weighted_tensor = (v * weights).sum(dim=0) / weights.sum()
                    if k.endswith("_input_ids") or k.endswith("_labels"):
                        weighted_tensor = weighted_tensor.long()
                    elif k.endswith("_attention_mask"):
                        weighted_tensor = weighted_tensor.clamp(0, 1)
                        # print(f"Env {env_id}, {k} max: {weighted_tensor.max().item()}, min: {weighted_tensor.min().item()}")
                    env_groups[env_id][k] = weighted_tensor.to(self.accelerator.device)
        
        return env_groups

    def concatenated_inputs(self, batch):
        # concatenate the chosen and rejected response
        rejected_max_len = max(
            [batch[key].shape[1] for key in batch if key.startswith("rejected") and key.endswith("_input_ids")])
        max_length = max(batch["chosen_input_ids"].shape[1], rejected_max_len)
        concatenated_batch = {}
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                # concatenated_key = k.replace("rejected", "concatenated")
                prefix = k.split("_")[0]
                concatenated_key = "concatenated" + k[len(prefix):]
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device)
        return concatenated_batch



    def concatenated_forward(self, model, batch):
        concatenated_batch = self.concatenated_inputs(batch)
        batch_size, seq_length = concatenated_batch["concatenated_input_ids"].shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.accelerator.device).unsqueeze(0).expand(batch_size, -1)

        attention_mask = concatenated_batch["concatenated_attention_mask"]
        if attention_mask.dtype != torch.bool:
            attention_mask = (attention_mask == 1).to(self.accelerator.device)
            # print(f"Converted attention_mask to bool, shape: {attention_mask.shape}")
        
        outputs = model(
            input_ids=concatenated_batch["concatenated_input_ids"],
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        all_logits = outputs.logits.to(torch.float32)
        all_logps = self._get_batch_logps(all_logits, concatenated_batch["concatenated_labels"])
        batch_size = batch["chosen_input_ids"].shape[0]
        chosen_logps = all_logps[:batch_size]
        rejected_logps = {}
        cnt = 0
        for k in batch:
            if k.startswith("rejected") and k.endswith("_input_ids"):
                cnt += 1
                rejected_logps[f"rejected{cnt}"] = all_logps[batch_size * cnt:batch_size * (cnt + 1)]
        return chosen_logps, rejected_logps, all_logits[:batch_size], {
            k: all_logits[batch_size * i:batch_size * (i + 1)] for i, k in enumerate(rejected_logps, 1)
        }

    def dpo_loss(self, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps):

        # Calculate the ratio of log probabilities
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = {
            k: policy_rejected_logps[k] - reference_rejected_logps[k] 
            for k in policy_rejected_logps
        }

        # Add numerical stability handling: limit the range of log ratios
        clamp_min, clamp_max = -10.0, 10.0  # Tunable parameter with limited range
        chosen_logratios = torch.clamp(chosen_logratios, min=clamp_min, max=clamp_max)
        rejected_logratios = {
            k: torch.clamp(rejected_logratios[k], min=clamp_min, max=clamp_max) 
            for k in rejected_logratios
        }

        temp = sum(
            torch.exp(self.beta * (rejected_logratios[k] - chosen_logratios)) 
            for k in rejected_logratios
        )
        
        #  Ensure temp is non-zero to avoid log(0)
        temp = torch.clamp(temp, min=1e-10)  # Add a small positive number to prevent log(0)
        losses = torch.log(1+temp)

        # Compute the reward function loss
        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = {
            k: self.beta * rejected_logratios[k].detach() 
            for k in rejected_logratios
        }

        return losses, chosen_rewards, rejected_rewards

    # def dpo_loss(self, policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps):
    #
    #     # Compute log probability ratios
    #     chosen_logratios = policy_chosen_logps - reference_chosen_logps
    #     rejected_logratios = {
    #         k: policy_rejected_logps[k] - reference_rejected_logps[k]
    #         for k in policy_rejected_logps
    #     }
    #
    #     # Add numerical stability handling: clamp log ratio values
    #     clamp_min, clamp_max = -10.0, 10.0  # Tunable parameter with limited range
    #     chosen_logratios = torch.clamp(chosen_logratios, min=clamp_min, max=clamp_max)
    #     rejected_logratios = {
    #         k: torch.clamp(rejected_logratios[k], min=clamp_min, max=clamp_max)
    #         for k in rejected_logratios
    #     }
    #
    #     # Compute loss
    #     # Using exp is safer after clamping, significantly reducing overflow risk
    #     temp = sum(
    #         torch.exp(self.beta * (rejected_logratios[k] - chosen_logratios))
    #         for k in rejected_logratios
    #     )
    #
    #     # Ensure temp is non-zero to avoid log(0)
    #     temp = torch.clamp(temp, min=1e-10)  # Add epsilon to prevent log(0)
    #     losses = torch.log(1 + temp)
    #
    #     # Compute rewards (for monitoring)
    #     chosen_rewards = self.beta * chosen_logratios.detach()
    #     rejected_rewards = {
    #         k: self.beta * rejected_logratios[k].detach()
    #         for k in rejected_logratios
    #     }
    #
    #     return losses, chosen_rewards, rejected_rewards

    def _get_batch_logps(self, logits, labels, average_log_prob=False):
        # Ensure  labels is the torch.long
        labels = labels[:, 1:].clone().long()

        if logits.device != labels.device:
            labels = labels.to(logits.device)

        # Check the range of labels
        vocab_size = logits.shape[-1]
        if labels.max() >= vocab_size or labels.min() < 0:
            # replace the neggative value to zero（如 label_pad_token_id）替换为 0
            labels[labels < 0] = 0
            # print(f"Replaced negative values in labels with 0.")

        # Check the range of  labels again
        if labels.max() >= vocab_size:
            raise ValueError(f"labels contains out-of-range values: max={labels.max()}, vocab_size={vocab_size}")

        # Process the label_pad_token_id
        loss_mask = labels != self.label_pad_token_id
        labels[labels == self.label_pad_token_id] = 0  # Mask padding with 0

        # Compute log probabilities per token
        log_probs = logits[:, :-1, :].log_softmax(-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)

        # Return average log probability or sum
        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)


    def compute_mmd(self, policy_outputs: dict[int, torch.Tensor], kernel: str = "rbf",
                    sigma: float = 1.0) -> torch.Tensor:
        """#Compute MMD of policy model across different groups"""
        def rbf_kernel(x1, x2, sigma):
            diff = x1.unsqueeze(1) - x2.unsqueeze(0)
            return torch.exp(-torch.sum(diff ** 2, dim=-1) / (2 * sigma ** 2))

        mmd_sum = 0.0
        num_groups = len(policy_outputs)
        group_pairs = [(i, j) for i in range(num_groups) for j in range(i + 1, num_groups)]

        for i, j in group_pairs:
            policy_i, policy_j = policy_outputs.get(i, torch.tensor([])), policy_outputs.get(j, torch.tensor([]))

            if policy_i.size(0) == 0 or policy_j.size(0) == 0:
                continue

            k_ii = rbf_kernel(policy_i, policy_i, sigma).mean()
            k_jj = rbf_kernel(policy_j, policy_j, sigma).mean()
            k_ij = rbf_kernel(policy_i, policy_j, sigma).mean()
            mmd = k_ii + k_jj - 2 * k_ij

            mmd_sum += mmd

        return mmd_sum / max(1, len(group_pairs)) if group_pairs else torch.tensor(0.0).to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False):
        if not self.use_dpo_data_collator:
            warnings.warn("compute_loss expects DPODataCollatorWithPadding.", UserWarning)

        # Soft clustering driven by causality
        cluster_probs = self.cluster_batch(inputs)
        env_groups = self.group_by_env(inputs, cluster_probs)

        total_dpo_loss = 0.0
        metrics = {}
        policy_outputs = {}

        # Compute DPO loss and outputs per environment
        for env_id, env_batch in env_groups.items():
            policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, _ = self.concatenated_forward(model,
                                                                                                            env_batch)
            with torch.no_grad():
                ref_chosen_logps, ref_rejected_logps, _, _ = self.concatenated_forward(self.ref_model, env_batch)

            dpo_losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps
            )
            total_dpo_loss += dpo_losses.mean()


            policy_outputs[env_id] = policy_chosen_logits.mean(dim=1)

            # Record the metric
            for key in rejected_rewards:
                margins = (chosen_rewards - rejected_rewards[key]).cpu().numpy().mean()

            prefix = ""
            metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().numpy().mean()
            for key in rejected_rewards:
                metrics[f"{prefix}rewards/{key}"] = rejected_rewards[key].cpu().numpy().mean()
            for key in rejected_rewards:
                metrics[f"{prefix}rewards/margins-{key}"] = margins


        total_dpo_loss /= len(env_groups)  # average DPO loss
        # computer the MMD item regulation
        mmd_loss = self.compute_mmd(policy_outputs, kernel=self.mmd_kernel, sigma=self.mmd_sigma)

        # total loss
        total_loss = total_dpo_loss + self.mmd_lambda * mmd_loss

        if self.accelerator.is_main_process:
            metrics["mmd_loss"] = mmd_loss.item()
            self.store_metrics(metrics)

        return (total_loss, metrics) if return_outputs else total_loss

    def store_metrics(self, metrics, train_eval="train"):
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Perform a prediction step for evaluation.
        Args:
            model: The model to evaluate.
            inputs: Input dictionary from the dataloader.
            prediction_loss_only: If True, only return the loss.
            ignore_keys: Keys to ignore in the output.
        Returns:
            Tuple of (loss, logits, labels) or just loss if prediction_loss_only=True.
        """
        inputs = self._prepare_inputs(inputs)
        
        with torch.no_grad():

            if prediction_loss_only:
                loss = self.compute_loss(model, inputs, return_outputs=False)
                return (loss, None, None)
            else:
                loss, metrics = self.compute_loss(model, inputs, return_outputs=True)

                policy_chosen_logps, policy_rejected_logps, policy_chosen_logits, _ = self.concatenated_forward(model, inputs)
                labels = inputs.get("concatenated_labels", None)
                return (loss, policy_chosen_logits, labels)


    def log(self, logs) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)




