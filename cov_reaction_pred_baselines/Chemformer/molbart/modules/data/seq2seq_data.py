""" Module containing classes to load seq2seq data"""
from typing import Any, Dict, List, Tuple

import pandas as pd
from rdkit import Chem

import torch
from molbart.modules.data.base import SimpleReactionListDataModule


class InMemorySynthesisDataModule(SimpleReactionListDataModule):
    """
    DataModule for lists of reactants / products. Used for in-memory forward/backward
    synthesis inference. This allows for updating the data while keeping the same
    global Chemformer model. Used e.g. in the FastAPI Chemformer-service.
    """

    def __init__(
        self,
        reactants: List[str],
        products: List[str],
        augment_prob: float = 0.0,
        reverse: bool = False,
        dataset_path="",
        **kwargs
    ):
        super().__init__(augment_prob, reverse, dataset_path=dataset_path, **kwargs)

        self._all_data = {"reactants": reactants, "products": products}

    def _load_all_data(self) -> None:
        pass


class SynthesisDataModule(SimpleReactionListDataModule):
    """
    DataModule for forward and backard synthesis prediction.

    The reactions are read from a tab seperated DataFrame .csv file.
    Expects the dataset to contain SMILES in two seperate columns named "reactants" and "products".
    The dataset must also contain a columns named "set" with values of "train", "val" and "test".
    validation column can be named "val", "valid" or "validation".

    All rows that are not test or validation, are assumed to be training samples.
    """

    def _get_sequences(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[List[str], List[str]]:
        reactants = [item["reactants"] for item in batch]
        products = [item["products"] for item in batch]
        if train:
            reactants = self._batch_augmenter(reactants)
            products = self._batch_augmenter(products)
        return reactants, products

    def _load_all_data(self) -> None:
        df = pd.read_csv(self.dataset_path, sep="\t").reset_index()
        self._all_data = {
            "reactants": df["reactants"].tolist(),
            "products": df["products"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)

class CovDockerSynthesisDataModule(SimpleReactionListDataModule):
    """
    DataModule for forward and backard synthesis prediction.

    The reactions are read from a tab seperated DataFrame .csv file.
    Expects the dataset to contain SMILES in two seperate columns named "reactants" and "products".
    The dataset must also contain a columns named "set" with values of "train", "val" and "test".
    validation column can be named "val", "valid" or "validation".

    All rows that are not test or validation, are assumed to be training samples.
    """

    def _get_sequences(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[List[str], List[str]]:
        reactants = [item["reactants"] for item in batch]
        products = [item["products"] for item in batch]
        if train:
            reactants = self._batch_augmenter(reactants)
            products = self._batch_augmenter(products)
        return reactants, products

    def _load_all_data(self) -> None:
        df = pd.read_csv(self.dataset_path).reset_index()
        df = df.dropna(subset=['reactants','products','set','src']).reset_index()
        self._all_data = {
            "reactants": df["reactants"].tolist(),
            "products": df["products"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)
    
    
    def _transform_batch(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        encoder_smiles, decoder_smiles = self._get_sequences(batch, train)
        encoder_ids, encoder_mask = self._encoder(
            encoder_smiles, add_sep_token=self.unified_model and not self.reverse
        )
        decoder_ids, decoder_mask = self._encoder(
            decoder_smiles, add_sep_token=self.unified_model and self.reverse
        )
        assert self.reverse == False, 'CovDocker reaction should be direct synthesis'
        if not self.reverse:
            return encoder_ids, encoder_mask, encoder_smiles, decoder_ids, decoder_mask, decoder_smiles
        return decoder_ids, decoder_mask, encoder_ids, encoder_mask, encoder_smiles

    def _collate(
        self, batch: List[Dict[str, Any]], train: bool = True
    ) -> Dict[str, Any]:
        (
            encoder_ids,
            encoder_mask,
            encoder_smiles,
            decoder_ids,
            decoder_mask,
            smiles,
        ) = self._transform_batch(batch, train)
        assert self.unified_model == False, 'Unified model not supported yet'
        if self.unified_model:
            return self._make_unified_model_batch(
                encoder_ids, encoder_mask, decoder_ids, decoder_mask, smiles
            )
        return {
            "encoder_input": encoder_ids,
            "encoder_pad_mask": encoder_mask,
            "encoder_smiles": encoder_smiles,
            "decoder_input": decoder_ids[:-1, :],
            "decoder_pad_mask": decoder_mask[:-1, :],
            "target": decoder_ids.clone()[1:, :],
            "target_mask": decoder_mask.clone()[1:, :],
            "target_smiles": smiles,
        }

class CovDockerConfidenceDataModule(SimpleReactionListDataModule):
    """
    DataModule for forward synthesis confidence prediction.

    The reactions are read from a "," seperated DataFrame .csv file.
    Expects the dataset to contain SMILES in two seperate columns named "reactants" and "products".
    The dataset must also contain a columns named "set" with values of "train", "val" and "test".
    validation column can be named "val", "valid" or "validation".

    All rows that are not test or validation, are assumed to be training samples.
    """

    def _get_sequences(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[List[str], List[str]]:
        # rxn = [item["rxn"] for item in batch]
        confidence = [item['confidence'] for item in batch]
        reactants = [item["rxn"].split('>>')[0] for item in batch]
        products = [item["rxn"].split('>>')[1] for item in batch]
        if train:
            reactants = self._batch_augmenter(reactants)
            products = self._batch_augmenter(products)
        # rxn = [f"{r}>>{p}" for r,p in zip(reactants,products)]
        return products,reactants, confidence

    def _load_all_data(self) -> None:
        df = pd.read_csv(self.dataset_path).reset_index()
        self._all_data = {
            "rxn": df['rxn'].tolist(),
            "confidence": df['confidence'].tolist(),
        }
        self._set_split_indices_from_dataframe(df)
    
    
    def _transform_batch(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        products, reactants, target = self._get_sequences(batch, train)
        reactants_ids, reactants_mask = self._encoder(
            reactants, add_sep_token=self.unified_model and not self.reverse
        )
        products_ids, products_mask = self._encoder(
            products, add_sep_token=self.unified_model and not self.reverse
        )
        target = torch.Tensor(target, device = reactants_ids.device)
        return reactants_ids, reactants_mask, reactants, products_ids, products_mask, products, target

    def _collate(
        self, batch: List[Dict[str, Any]], train: bool = True
    ) -> Dict[str, Any]:
        (
            reactants_ids, 
            reactants_mask, 
            reactants_smiles, 
            products_ids, 
            products_mask, 
            products_smiles, 
            target
        ) = self._transform_batch(batch, train)
        assert self.unified_model == False, 'Unified model not supported yet'
        return {
            "reactants_input": reactants_ids,
            "reactants_pad_mask": reactants_mask,
            "reactants_smiles": reactants_smiles,
            "products_input": products_ids,
            "products_pad_mask": products_mask,
            "products_smiles": products_smiles,
            "target": target,
        }




class Uspto50DataModule(SimpleReactionListDataModule):
    """
    DataModule for the USPTO-50 dataset

    The reactions as well as a type token are read from
    a pickled DataFrame
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._include_type_token = kwargs.get("include_type_token", False)

    def _get_sequences(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[List[str], List[str]]:
        reactants = [Chem.MolToSmiles(item["reactants"]) for item in batch]
        products = [Chem.MolToSmiles(item["products"]) for item in batch]

        if train:
            reactants = self._batch_augmenter(reactants)
            products = self._batch_augmenter(products)

        if self._include_type_token and not self.reverse:
            reactants = [
                item["type_tokens"] + smi for item, smi in zip(batch, reactants)
            ]
        if self._include_type_token and self.reverse:
            products = [item["type_tokens"] + smi for item, smi in zip(batch, products)]

        return reactants, products

    def _load_all_data(self) -> None:
        df = pd.read_pickle(self.dataset_path).reset_index()
        self._all_data = {
            "reactants": df["reactants_mol"].tolist(),
            "products": df["products_mol"].tolist(),
            "type_tokens": df["reaction_type"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)


class UsptoMixedDataModule(SimpleReactionListDataModule):
    """
    DataModule for the USPTO-Mixed dataset

    The reactions are read from a pickled DataFrame
    """

    def _get_sequences(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[List[str], List[str]]:
        reactants = [Chem.MolToSmiles(item["reactants"]) for item in batch]
        products = [Chem.MolToSmiles(item["products"]) for item in batch]
        if train:
            reactants = self._batch_augmenter(reactants)
            products = self._batch_augmenter(products)
        return reactants, products

    def _load_all_data(self) -> None:
        df = pd.read_pickle(self.dataset_path).reset_index()
        self._all_data = {
            "reactants": df["reactants_mol"].tolist(),
            "products": df["products_mol"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)


class UsptoSepDataModule(SimpleReactionListDataModule):
    """
    DataModule for the USPTO-Separated dataset

    The reactants, reagents and products are read from
    a pickled DataFrame
    """

    def _get_sequences(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[List[str], List[str]]:
        reactants = [Chem.MolToSmiles(item["reactants"]) for item in batch]
        reagents = [Chem.MolToSmiles(item["reagents"]) for item in batch]
        products = [Chem.MolToSmiles(item["products"]) for item in batch]

        if train:
            reactants = self._batch_augmenter(reactants)
            reagents = self._batch_augmenter(reagents)
            products = self._batch_augmenter(products)

        reactants = [
            react_smi + ">" + reag_smi
            for react_smi, reag_smi in zip(reactants, reagents)
        ]

        return reactants, products

    def _load_all_data(self) -> None:
        df = pd.read_pickle(self.dataset_path).reset_index()
        self._all_data = {
            "reactants": df["reactants_mol"].tolist(),
            "products": df["products_mol"].tolist(),
            "reagents": df["reagents_mol"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)


class MolOptDataModule(SimpleReactionListDataModule):
    """
    DataModule for a dataset for molecular optimization

    The input and ouput molecules, as well as a the property
    tokens are read from a pickled DataFrame
    """

    def _get_sequences(
        self, batch: List[Dict[str, Any]], train: bool
    ) -> Tuple[List[str], List[str]]:
        input_smiles = [Chem.MolToSmiles(item["input_mols"]) for item in batch]
        output_smiles = [Chem.MolToSmiles(item["output_mols"]) for item in batch]

        if train:
            input_smiles = self._batch_augmenter(input_smiles)
            output_smiles = self._batch_augmenter(output_smiles)

        input_smiles = [
            item["prop_tokens"] + smi for item, smi in zip(batch, input_smiles)
        ]

        return input_smiles, output_smiles

    def _load_all_data(self) -> None:
        df = pd.read_pickle(self.dataset_path).reset_index()
        self._all_data = {
            "prop_tokens": df["property_tokens"].tolist(),
            "input_mols": df["input_mols"].tolist(),
            "output_mols": df["output_mols"].tolist(),
        }
        self._set_split_indices_from_dataframe(df)
