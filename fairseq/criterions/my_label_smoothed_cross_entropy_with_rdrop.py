import logging
import math
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import Tensor

from fairseq import utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from fairseq.criterions.label_smoothed_cross_entropy_with_rdrop import (
    RdropLabelSmoothedCrossEntropyCriterionConfig,
    RdropLabelSmoothedCrossEntropyCriterion
)
from fairseq.logging import metrics
from fairseq.modules.fairseq_dropout import FairseqDropout

logger = logging.getLogger(__name__)


@dataclass
class MyRdropLabelSmoothedCrossEntropyCriterionConfig(RdropLabelSmoothedCrossEntropyCriterionConfig):
    rdrop_n: int = field(
        default=2,
        metadata={"help": ""},
    )
    rdrop_type: str = field(
        default="js",
        metadata={"help": ""},
    )
    rdrop_step: int = field(
        default=1,
        metadata={"help": ""},
    )

    freeze_encoder_rdrop: bool = field(
        default=False, metadata={"help": ""}
    )
    freeze_decoder_rdrop: bool = field(
        default=False, metadata={"help": ""}
    )

    dependency_child_loss_alpha: float = field(
        default=0.,
        metadata={"help": ""}
    )
    dependency_parent_loss_beta: float = field(
        default=0.,
        metadata={"help": ""}
    )
    dependency_attn_layers: str = field(
        default="",
        metadata={"help": ""}
    )
    solid_dp: bool = field(
        default=False, metadata={"help": ""}
    )

    contrastive_lambda: float = field(
        default=0.,
        metadata={"help": ""}
    )
    contrastive_temperature: float = field(
        default=1.,
        metadata={"help": ""}
    )


@register_criterion(
    "my_label_smoothed_cross_entropy_with_rdrop",
    dataclass=MyRdropLabelSmoothedCrossEntropyCriterionConfig,
)
class MyRdropLabelSmoothedCrossEntropyCriterion(RdropLabelSmoothedCrossEntropyCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            rdrop_alpha: float = 0.,
            rdrop_n: int = 2,
            rdrop_type: str = "kl",
            rdrop_step: int = 1,
            freeze_encoder_rdrop: bool = False,
            freeze_decoder_rdrop: bool = False,
            dependency_child_loss_alpha: float = 0.,
            dependency_parent_loss_beta: float = 0.,
            dependency_attn_layers: str = "",
            solid_dp: bool = False,
            contrastive_lambda: float = 0.,
            contrastive_temperature: float = 1.,
    ):
        super().__init__(
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=ignore_prefix_size,
            report_accuracy=report_accuracy,
            rdrop_alpha=rdrop_alpha
        )
        self.rdrop_n = rdrop_n if rdrop_n >= 2 else 1
        self.rdrop_type = rdrop_type
        self.rdrop_step = rdrop_step if rdrop_step > 1 else 1

        self.freeze_encoder_rdrop = freeze_encoder_rdrop
        self.freeze_decoder_rdrop = freeze_decoder_rdrop
        self.register_hook = False

        self.dependency_child_loss_alpha = dependency_child_loss_alpha
        self.dependency_parent_loss_beta = dependency_parent_loss_beta
        if not dependency_attn_layers:
            dependency_attn_layers = "-1"
        self.dependency_attn_layers = [int(x) for x in dependency_attn_layers.split(",")]
        self.solid_dp = solid_dp

        self.contrastive_lambda = contrastive_lambda
        self.contrastive_temperature = contrastive_temperature

    def forward(self, model, sample, reduce=True, net_output=None, use_rdrop: bool = True,
                load_dependency: bool = False, empty_cache: bool = False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_concat_input = None
        if use_rdrop and self.rdrop_alpha > 0 and model.get_num_updates() % self.rdrop_step == 0 \
                and sample["net_input"]["src_tokens"].size(0) == sample["target"].size(0):
            if self.freeze_encoder_rdrop and self.freeze_decoder_rdrop:  # 不用rdrop
                net_output = model(**sample["net_input"])
            elif not self.freeze_encoder_rdrop and self.freeze_decoder_rdrop:  # 在encoder上用rdrop
                sample_concat_input = self.duplicate_sample_input(sample["net_input"])
                net_output = self.get_net_output_using_rdrop_only_encoder(model, sample, sample_concat_input)
            elif self.freeze_encoder_rdrop and not self.freeze_decoder_rdrop:  # 在decoder上用rdrop
                sample_concat_input = self.duplicate_sample_input(sample["net_input"])
                net_output = self.get_net_output_using_rdrop_only_decoder(model, sample, sample_concat_input)
            else:  # 用rdrop
                sample_concat_input = self.duplicate_sample_input(sample["net_input"])
                net_output = model(**sample_concat_input)
        else:
            net_output = model(**sample["net_input"], solid_dp=self.solid_dp, dp_layers=self.dependency_attn_layers)

        loss, nll_loss, rdrop_kl_loss, dependency_child_loss, dependency_parent_loss, contrastive_loss = \
            self.compute_loss(
                model, net_output, sample, reduce, load_dependency=load_dependency, sample_input=sample_concat_input,
                use_rdrop=use_rdrop, empty_cache=empty_cache
            )

        sample_size = (sample["target"].size(0) if self.sentence_avg else sample["ntokens"])
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        logging_output["rdrop_kl_loss"] = 0
        logging_output["dependency_child_loss"] = 0
        logging_output["dependency_parent_loss"] = 0
        if use_rdrop and self.rdrop_alpha > 0:
            logging_output["rdrop_kl_loss"] = utils.item(rdrop_kl_loss.data)
        if load_dependency:
            if dependency_child_loss is not None:
                logging_output["dependency_child_loss"] = utils.item(dependency_child_loss.data)
            if dependency_parent_loss is not None:
                logging_output["dependency_parent_loss"] = utils.item(dependency_parent_loss.data)
        if self.contrastive_lambda > 0. and model.training:
            logging_output["contrastive_loss"] = utils.item(contrastive_loss.data)
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True, load_dependency: bool = False, sample_input=None,
                     use_rdrop: bool = True, empty_cache: bool = False):
        if sample_input is None:
            sample_input = sample["net_input"]

        lprobs, target = self.get_lprobs_and_target(model, net_output, sample, use_rdrop=use_rdrop)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        rdrop_kl_loss = None
        if use_rdrop and self.rdrop_alpha > 0:
            pad_mask = target[: target.size(0) // self.rdrop_n].unsqueeze(-1).eq(self.padding_idx)
            if self.rdrop_type == "js":
                rdrop_kl_loss = self.compute_js_loss(model, net_output, pad_mask)
            else:
                rdrop_kl_loss = self.compute_kl_loss(model, net_output, pad_mask)
            loss += self.rdrop_alpha * rdrop_kl_loss
            if empty_cache:
                torch.cuda.empty_cache()

        dependency_child_loss = None
        dependency_parent_loss = None
        if load_dependency and self.dependency_child_loss_alpha > 0.:
            dependency_child_loss = self.calc_dependency_attn_loss(
                sample_input["src_dep_child_matrix"],
                [net_output[1]["child_attn_head"][i] for i in self.dependency_attn_layers],
                divide_rdrop_n=use_rdrop
            )
            loss += self.dependency_child_loss_alpha * dependency_child_loss
        if load_dependency and self.dependency_parent_loss_beta > 0.:
            dependency_parent_loss = self.calc_dependency_attn_loss(
                sample_input["src_dep_parent_matrix"],
                [net_output[1]["parent_attn_head"][i] for i in self.dependency_attn_layers],
                divide_rdrop_n=use_rdrop
            )
            loss += self.dependency_parent_loss_beta * dependency_parent_loss
            if empty_cache:
                torch.cuda.empty_cache()

        contrastive_loss = None
        if self.contrastive_lambda > 0. and model.training:
            contrastive_loss = self.interleaved_contrastive_loss(
                net_output[1]["encoder_out"][0].transpose(0, 1),
                sample_input["src_tokens"]
            )
            loss += self.contrastive_lambda * contrastive_loss
            if empty_cache:
                torch.cuda.empty_cache()

        return loss, nll_loss, rdrop_kl_loss, dependency_child_loss, dependency_parent_loss, contrastive_loss

    def get_lprobs_and_target(self, model, net_output, sample, use_rdrop: bool = True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if use_rdrop and self.rdrop_alpha > 0 or target.size(0) != lprobs.size(0):
            tensor_list = [target]
            for _ in range(1, self.rdrop_n):
                tensor_list.append(target.clone())
            target = torch.cat(tensor_list, dim=0)

        if self.ignore_prefix_size > 0:
            # lprobs: B x T x C
            lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_kl_loss(self, model, net_output, pad_mask=None, reduce=True):
        net_prob = model.get_normalized_probs(net_output, log_probs=True)
        net_prob_tec = model.get_normalized_probs(net_output, log_probs=False)

        net_prob = net_prob.view(-1, net_prob.size(-1))
        net_prob_tec = net_prob_tec.view(-1, net_prob_tec.size(-1))

        p_list = torch.split(net_prob, net_prob.size(0) // self.rdrop_n, dim=0)
        p_tec_list = torch.split(net_prob_tec, net_prob_tec.size(0) // self.rdrop_n, dim=0)

        p_loss_list = []
        for i in range(self.rdrop_n):
            for j in range(self.rdrop_n):
                if i == j:
                    continue
                p_loss = F.kl_div(p_list[i], p_tec_list[j], reduction="none")
                if pad_mask is not None:
                    p_loss.masked_fill_(pad_mask, 0.0)
                if reduce:
                    p_loss = p_loss.sum()
                p_loss_list.append(p_loss)

        loss = sum(p_loss_list) / len(p_loss_list)
        return loss

    def compute_js_loss(self, model, net_output, pad_mask=None, reduce=True):
        net_prob = model.get_normalized_probs(net_output, log_probs=True)
        net_prob_tec = model.get_normalized_probs(net_output, log_probs=False)

        net_prob = net_prob.view(-1, net_prob.size(-1))
        net_prob_tec = net_prob_tec.view(-1, net_prob_tec.size(-1))

        p_list = torch.split(net_prob, net_prob.size(0) // self.rdrop_n, dim=0)
        p_tec_list = torch.split(net_prob_tec, net_prob_tec.size(0) // self.rdrop_n, dim=0)
        m_tec = sum(p_tec_list) / self.rdrop_n

        p_loss_list = []
        for i in range(self.rdrop_n):
            p_loss = F.kl_div(p_list[i], m_tec, reduction="none")
            if pad_mask is not None:
                p_loss.masked_fill_(pad_mask, 0.0)
            if reduce:
                p_loss = p_loss.sum()
            p_loss_list.append(p_loss)

        loss = sum(p_loss_list) / len(p_loss_list)
        return loss

    def duplicate_sample_input(self, sample_input):
        sample_concat_input = dict()
        for k, v in sample_input.items():
            if k in ["src_tokens", "src_lengths", "prev_output_tokens",
                     "src_dep", "src_dep_parent_matrix", "src_dep_child_matrix", "src_dep_mask_matrix"]:
                tensor_list = [v]
                for i in range(1, self.rdrop_n):
                    tensor_list.append(v.clone())
                sample_concat_input[k] = torch.cat(tensor_list, dim=0)
        return sample_concat_input

    def get_net_output_using_rdrop_only_encoder(self, model, sample, sample_concat_input=None):
        if sample_concat_input is None:
            sample_concat_input = self.duplicate_sample_input(sample["net_input"])
        encoder_out = model.encoder(
            sample_concat_input["src_tokens"],
            src_lengths=sample_concat_input["src_lengths"],
            return_all_hiddens=True
        )

        def dropout_forward_hook(module, input):
            return module.forward_mask_fixed(*input)

        if not self.register_hook:
            for name, submodule in model.decoder.named_modules():
                if any(isinstance(submodule, cls) for cls in [FairseqDropout]):
                    logger.info(f"Dropout found in '{name}', register hook.")
                    submodule.register_forward_pre_hook(dropout_forward_hook)
            self.register_hook = True

        encoder_out_list = self.split_encoder_out(encoder_out)
        decoder_out_list = list()
        for encoder_out_part in encoder_out_list:
            decoder_out_part = model.decoder(
                sample["net_input"]["prev_output_tokens"],
                encoder_out=encoder_out_part,
                features_only=False,
                alignment_layer=None,
                alignment_heads=None,
                src_lengths=sample["net_input"]["src_lengths"],
                return_all_hiddens=True,
            )
            decoder_out_list.append(decoder_out_part)

        for name, submodule in model.decoder.named_modules():
            if any(isinstance(submodule, cls) for cls in [FairseqDropout]):
                submodule.reset_dropout_mask()

        decoder_out = self.join_decoder_out(decoder_out_list)
        return decoder_out

    def split_encoder_out(self, encoder_out):
        encoder_out_list_result = list()
        for i in range(self.rdrop_n):
            encoder_out_list_result.append(dict())
            for k in ["fc_results", "src_tokens"]:
                encoder_out_list_result[i][k] = encoder_out[k]
            for k in ["encoder_padding_mask", "encoder_embedding", "src_lengths", "encoder_out", "encoder_states"]:
                encoder_out_list_result[i][k] = list()

        for k in ["encoder_padding_mask", "encoder_embedding", "src_lengths", "parent_attn_head", "child_attn_head"]:
            for v_tensor in encoder_out[k]:
                v_tensor_parts = torch.chunk(v_tensor, self.rdrop_n, dim=0)
                for i in range(self.rdrop_n):
                    encoder_out_list_result[i][k].append(v_tensor_parts[i])
        for k in ["encoder_out", "encoder_states"]:
            for v_tensor in encoder_out[k]:
                v_tensor_parts = torch.chunk(v_tensor, self.rdrop_n, dim=1)
                for i in range(self.rdrop_n):
                    encoder_out_list_result[i][k].append(v_tensor_parts[i])

        return encoder_out_list_result

    def join_decoder_out(self, decoder_out_list):
        decoder_out_result = [torch.cat([decoder_out[0] for decoder_out in decoder_out_list]), dict()]
        for k in ["attn", "inner_states"]:
            decoder_out_result[1][k] = list()
            for i in range(len(decoder_out_list[0][1][k])):
                decoder_out_result[1][k].append(torch.cat([decoder_out[1][k][i] for decoder_out in decoder_out_list],
                                                          dim=0 if k == "attn" else 1))

        return tuple(decoder_out_result)

    def get_net_output_using_rdrop_only_decoder(self, model, sample, sample_concat_input=None):
        if sample_concat_input is None:
            sample_concat_input = self.duplicate_sample_input(sample["net_input"])
        encoder_out = model.encoder(
            sample["net_input"]["src_tokens"],
            src_lengths=sample["net_input"]["src_lengths"],
            return_all_hiddens=True
        )
        encoder_out = self.duplicate_encoder_out(encoder_out)
        decoder_out = model.decoder(
            sample_concat_input["prev_output_tokens"],
            encoder_out=encoder_out,
            features_only=False,
            alignment_layer=None,
            alignment_heads=None,
            src_lengths=sample_concat_input["src_lengths"],
            return_all_hiddens=True,
        )
        return decoder_out

    def duplicate_encoder_out(self, encoder_out):
        encoder_out_duplicated = dict()
        for k, v in encoder_out.items():
            if k in ["fc_results", "src_tokens"]:
                encoder_out_duplicated[k] = v
            else:
                encoder_out_duplicated[k] = []
                for v_tensor in v:
                    tensor_list = [v_tensor]
                    for i in range(1, self.rdrop_n):
                        tensor_list.append(v_tensor.clone())
                    if k in ["encoder_padding_mask", "encoder_embedding", "src_lengths",
                             "parent_attn_head", "child_attn_head"]:
                        encoder_out_duplicated[k].append(torch.cat(tensor_list, dim=0))
                    elif k in ["encoder_out", "encoder_states"]:
                        encoder_out_duplicated[k].append(torch.cat(tensor_list, dim=1))
                    else:
                        raise NotImplementedError

        return encoder_out_duplicated

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        super().reduce_metrics(logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        dependency_child_loss = utils.item(
            sum(log.get("dependency_child_loss", 0) for log in logging_outputs)
            / sample_size
            / math.log(2)
        )
        if not dependency_child_loss == 0.:
            metrics.log_scalar("dependency_child_loss", dependency_child_loss)

        dependency_parent_loss = utils.item(
            sum(log.get("dependency_parent_loss", 0) for log in logging_outputs)
            / sample_size
            / math.log(2)
        )
        if not dependency_parent_loss == 0.:
            metrics.log_scalar("dependency_parent_loss", dependency_parent_loss)

        contrastive_loss = utils.item(
            sum(log.get("contrastive_loss", 0) for log in logging_outputs)
            / sample_size
            / math.log(2)
        )
        if not contrastive_loss == 0.:
            metrics.log_scalar("contrastive_loss", contrastive_loss)

    def calc_dependency_attn_loss(self, dep_matrix: Optional[Tensor], attn_head_list: List[Tensor],
                                  avg: bool = False, divide_rdrop_n: bool = True):
        if dep_matrix is None or not attn_head_list or not dep_matrix.shape == attn_head_list[0].shape:
            raise RuntimeError

        loss_list = [-torch.sum(dep_matrix * torch.log(attn_head + 1e-9)) for attn_head in attn_head_list]

        loss = sum(loss_list)
        if divide_rdrop_n and self.rdrop_alpha > 0.:
            loss /= self.rdrop_n
        return loss / len(loss_list) if avg else loss

    def interleaved_contrastive_loss(self, encoder_out: Tensor, src_tokens: Tensor, pad_index: int = -1):
        # encoder_out: [batch_size, seq_len, hidden_size]
        # src_tokens: [batch_size, seq_len]
        if pad_index < 0:
            pad_index = self.padding_idx

        bsz, seq_len = encoder_out.shape[0], encoder_out.shape[1]
        ignore_pad_mask = torch.ones(bsz, seq_len).float()
        if pad_index != -1:
            ignore_pad_mask = (src_tokens != pad_index).float()

        snt_emb = torch.sum(encoder_out, dim=1) / torch.sum(ignore_pad_mask.float(), dim=1).unsqueeze(-1)

        indices = torch.div(torch.arange(bsz), 2, rounding_mode='floor')
        mask_matching = (indices[:, None] == indices[None, :]).float().to(snt_emb.device)

        mask_positive = mask_matching - torch.eye(bsz).to(snt_emb.device)
        mask_negative = torch.ones((bsz, bsz)).to(snt_emb.device) - mask_matching

        similarity_matrix = F.cosine_similarity(snt_emb.unsqueeze(1), snt_emb.unsqueeze(0), dim=-1)
        exp_sim_matrix = torch.exp(similarity_matrix / self.contrastive_temperature)
        exp_sim_vector_positive = torch.sum(exp_sim_matrix * mask_positive, dim=0)
        exp_sim_vector_negative = torch.sum(exp_sim_matrix * mask_negative, dim=0)
        loss = -torch.sum(torch.log(exp_sim_vector_positive / (exp_sim_vector_negative + 1e-9)))

        return seq_len * loss
