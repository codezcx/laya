# Trial-level emotion supervision v29

v9 only averaged predictions after window training. Test training at the trial
label's resolution, while giving both comparison arms identical inference
context. This changes the prediction unit; results cannot be presented as3s
online accuracy or directly ranked with earlier window results.

Usev28's fixed training subjects and nine FACED held videos, unchangedv26
per-channel spectral features. Group by(dataset,subject,recording path,trial),
reject conflicting labels/splits and repeated starts. Retain all valid windows
of each trial, including incomplete10-window bags due to the earlier quality
quarantine. Do not drop trials or add replacement windows.

Both arms use one shared StandardScaler fitted on training windows and the same
150-dimensional linear softmax architecture. Window-loss fits each window;
trial-loss fits the mean feature vector of each trial. Both predict from the
trial mean at validation/inference (equivalently mean linear logits). Thus the
controlled change is training loss granularity rather than inference context.

Use equal dataset/class weights at trial level. Divide a trial's window-loss
weight by its number of retained windows. Normalize total training weight to
the same number of training trials in both arms, preserving regularization scale.
Sweep C[.001,.01,.1,1,10],max_iter10,000, stop on warnings. Select by mean SEED and
FACED unseen-video subject-balanced trial accuracy; smallerC on exact ties.
Report FACED seen-video validation separately. No target access in this stage.

This is a fixed source-development diagnostic of weak window labels. It does not
establish text semantics, task recognition, or success on a fresh target. Source
development has been reused extensively; all differences need cautious treatment.
MI unchanged; no new tasks enabled.
