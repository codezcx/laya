import unittest
import torch
from eeg_align.task_options import load_tasks,make_question
from eeg_align.jev import choice_logits

class TaskOptionsTest(unittest.TestCase):
    def test_all_tasks_permutation_targets_and_label_independence(self):
        for task in load_tasks().values():
            seen=set()
            for epoch in range(30):
                orders=[]
                for c in task['classes']:
                    q=make_question(task,c['class_id'],'trial-1',epoch)
                    self.assertEqual(q['option_class_ids'][q['label']],c['class_id'])
                    self.assertEqual(q,make_question(task,c['class_id'],'trial-1',epoch))
                    orders.append(tuple(q['option_class_ids']))
                self.assertEqual(len(set(orders)),1)
                seen.update(orders)
            self.assertGreater(len(seen),1)

    def test_mixed_choice_mask_and_gradient(self):
        state=torch.randn(3,8,requires_grad=True)
        w=torch.randn(16,8,requires_grad=True);b=torch.randn(16,requires_grad=True)
        logits=choice_logits(state,w,b,[2,3,7])
        p=logits.softmax(-1)
        self.assertEqual(p[0,2:].sum().item(),0)
        self.assertEqual(p[1,3:].sum().item(),0)
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor([1,2,6]));loss.backward()
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertEqual(w.grad[7:].abs().sum().item(),0)
        torch.testing.assert_close(choice_logits(state,w,b,2),torch.nn.functional.linear(state,w[:2],b[:2]))
        with self.assertRaises(ValueError):choice_logits(state,w,b,[2,17,3])

if __name__=='__main__':unittest.main()

class TaskContrastTest(unittest.TestCase):
    def test_weighted_task_losses_and_gradients(self):
        from eeg_align.semantic_alignment import multipositive_loss,task_contrastive_loss
        torch.manual_seed(5)
        z=torch.randn(12,8,requires_grad=True);text=torch.randn(12,8)
        tasks=torch.tensor([0]*4+[1]*4+[2]*4)
        labels=torch.tensor([0,0,1,1,2,3,4,2,5,6,7,8])
        weights={0:.5,1:.25,2:.25}
        loss,parts=task_contrastive_loss(z,labels,tasks,text,weights,distributed=False)
        expected=sum(w*multipositive_loss(z[tasks==t],labels[tasks==t],text,distributed=False) for t,w in weights.items())
        torch.testing.assert_close(loss,expected)
        loss.backward();self.assertTrue(torch.isfinite(z.grad).all())
        self.assertTrue(all(z.grad[tasks==t].abs().sum()>0 for t in weights))
