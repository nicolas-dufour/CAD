import abc


class SyntheticCaptioner(abc.ABC):
    def create_captions(self, **kwargs):
        """
        Creates captions for the given images
        """
        raise NotImplementedError

    def caption_from_dataloader(self, dataloader):
        """
        Creates captions for the given images in a dataloader.
        """
        raise NotImplementedError

    def create_embeddings(self, captions):
        """
        Creates embeddings for the given captions
        """
        raise NotImplementedError

    def caption_and_embed(self, **kwargs):
        """
        Captions and embeds an image.
        """
        captions = self.create_captions(**kwargs)
        embedding = self.create_embeddings(captions)
        return captions, embedding
