from transformers import Trainer, is_torch_tpu_available
from typing import Dict

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm



class MoETrainer(Trainer):
    def __init__(self, loss_penalty,*args, **kwargs):
        super().__init__(*args,**kwargs)
        self.loss_penalty = loss_penalty

        self.CV = 0
        self.RC = 0
        self.RC_bal = 0
        self.steps = 0

    def compute_loss(self,model,inputs, return_outputs=False):
        lm_loss = super().compute_loss(model, inputs)
        if self.loss_penalty == 'fedus':
            lm_loss += model.fedus_loss(inputs)
        elif self.loss_penalty == 'loramoe':
            lm_loss += model.loramoe_loss(inputs)

        self.RC += model.RC(inputs)
        self.CV += model.CV(inputs)
        self.RC_bal += model.RC_bal(inputs)
        self.steps += 1
        
        return lm_loss
    
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ingore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_tpu_available():
                xm.mark_step()
            
            logs: Dict[str, float] = {}

            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            tr_loss -= tr_loss

            # Monitoring elements
            logs['loss'] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs['RC'] = self.RC / self.steps
            logs['RC_bal'] = self.RC_bal / self.steps
            logs['CV'] = self.CV / self.steps

            # Reset monitoring elements
            self.CV = 0
            self.RC = 0
            self.RC_bal = 0
            self.steps = 0

            if grad_norm is not None:
                logs['grad_norm'] = grad_norm
            logs['learning_rate'] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)
