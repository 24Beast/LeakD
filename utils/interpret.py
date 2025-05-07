import torch
from torch import Tensor
from torch.nn import Module
from captum.concept import TCAV, Concept
from typing import Union, Any, List, Dict
from captum.concept._utils.common import concepts_to_str


class ModifiedTCAV(TCAV):

    def __init__(
        self,
        model: Module,
        layers: Union[str, List[str]],
        model_id: str = "default_model_id",
        classifier=None,
        layer_attr_method=None,
        attribute_to_layer_input: bool = False,
        save_path: str = "./cav/",
        **classifier_kwargs: Any,
    ) -> None:
        r"""
        Args:

            model (Module): An instance of pytorch model that is used to compute
                    layer activations and attributions.
            layers (str or list[str]): A list of layer name(s) that are
                    used for computing concept activations (cavs) and layer
                    attributions.
            model_id (str, optional): A unique identifier for the PyTorch `model`
                    passed as first argument to the constructor of TCAV class. It
                    is used to store and load activations for given input `model`
                    and associated `layers`.
            classifier (Classifier, optional): A custom classifier class, such as the
                    Sklearn "linear_model" that allows us to train a model
                    using the activation vectors extracted for a layer per concept.
                    It also allows us to access trained weights of the model
                    and the list of prediction classes.
            layer_attr_method (LayerAttribution, optional): An instance of a layer
                    attribution algorithm that helps us to compute model prediction
                    sensitivity scores.

                    Default: None
                    If `layer_attr_method` is None, we default it to gradients
                    for the layers using `LayerGradientXActivation` layer
                    attribution algorithm.
            save_path (str, optional): The path for storing CAVs and
                    Activation Vectors (AVs).
            classifier_kwargs (Any, optional): Additional arguments such as
                    `test_split_ratio` that are passed to concept `classifier`.

        Examples::
            >>>
            >>> # TCAV use example:
            >>>
            >>> # Define the concepts
            >>> stripes = Concept(0, "stripes", striped_data_iter)
            >>> random = Concept(1, "random", random_data_iter)
            >>>
            >>>
            >>> mytcav = TCAV(model=imagenet,
            >>>     layers=['inception4c', 'inception4d'])
            >>>
            >>> scores = mytcav.interpret(inputs, [[stripes, random]], target = 0)
            >>>
            For more thorough examples, please check out TCAV tutorial and test cases.
        """

        TCAV.__init__(
            self,
            model,
            layers,
            model_id,
            classifier,
            layer_attr_method,
            **classifier_kwargs,
        )

    def _tcav_sub_computation(
        self,
        scores: Dict[str, Dict[str, Dict[str, Tensor]]],
        layer: str,
        attribs: Tensor,
        cavs: Tensor,
        classes: List[List[int]],
        experimental_sets: List[List[Concept]],
    ) -> None:
        # n_inputs x n_concepts
        tcav_score = torch.matmul(attribs.float(), torch.transpose(cavs, 1, 2))
        assert len(tcav_score.shape) == 3, (
            "tcav_score should have 3 dimensions: n_experiments x "
            "n_inputs x n_concepts."
        )

        assert attribs.shape[0] == tcav_score.shape[1], (
            "attrib and tcav_score should have the same 1st and "
            "2nd dimensions respectively (n_inputs)."
        )
        # n_experiments x n_concepts
        sign_count_score = torch.mean((tcav_score > 0.0).float(), dim=1)

        magnitude_score = torch.mean(tcav_score, dim=1)

        abs_magnitude_score = torch.mean(torch.abs(tcav_score), dim=1)

        for i, (cls_set, concepts) in enumerate(zip(classes, experimental_sets)):
            concepts_key = concepts_to_str(concepts)

            # sort classes / concepts in the order specified in concept_keys
            concept_ord = [concept.id for concept in concepts]
            class_ord = {cls_: idx for idx, cls_ in enumerate(cls_set)}

            new_ord = torch.tensor(
                [class_ord[cncpt] for cncpt in concept_ord], device=tcav_score.device
            )

            # sort based on classes
            scores[concepts_key][layer] = {
                "sign_count": torch.index_select(
                    sign_count_score[i, :], dim=0, index=new_ord
                ),
                "magnitude": torch.index_select(
                    magnitude_score[i, :], dim=0, index=new_ord
                ),
                "abs_magnitude": torch.index_select(
                    abs_magnitude_score[i, :], dim=0, index=new_ord
                ),
            }
