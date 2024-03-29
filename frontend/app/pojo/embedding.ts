import {Modality} from "~/pojo/modality";

export type Embedding = {
    id?: number;
    modalities: Modality[];
    deleted?: boolean;
}