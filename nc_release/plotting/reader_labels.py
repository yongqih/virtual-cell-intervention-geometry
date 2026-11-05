import matplotlib.text
labels={
 'Training identities':'Training interventions',
 'Calibration identities (cross-fit query blocks)':'Calibration interventions',
 'RPE1 cell halves':'RPE1 cell half-samples',
 'Frozen model':'Original model',
 'Frozen':'Original',
 'Shared − frozen':'Shared − original',
 'Measured − frozen':'Measured − original',
 'Frozen-case repair and newly introduced inversions':'Repair of original cases and new inversions',
 'All 26 frozen cases':'All 26 original cases',
 'Frozen, paired':'Original, paired',
 'Frozen, training':'Original, training',
 'Complete output folds':'Folds with all training outputs',
 'OBS71 descriptors':'Observational features',
 'Inner OOF':'Cross-validation',
 'Historical entry comparisons':'Measured entry comparisons',
 'Same halves':'Shared cell half-samples',
 'Independent halves':'Independent cell half-samples',
}
original_set=matplotlib.text.Text.set_text
def set_text(self,value):
    return original_set(self,labels.get(value,value) if isinstance(value,str) else value)
matplotlib.text.Text.set_text=set_text
