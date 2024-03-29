import {Encoder} from "~/pojo/encoder";
import {Modals} from "~/pojo/modals";

export type Modality = {
    encoder?: Encoder;
    modals?: Modals[];
    weight?: number;
}
