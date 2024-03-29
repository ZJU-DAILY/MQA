import {Modals} from "~/pojo/modals";

export type Dataset = {
    name: string;
    objects: number;
    modal: Modals[];
}

export const datasets: Dataset[] = [
    {
        name: 'MitStates',
        objects: 53743,
        modal: [
            {id: '0', name: 'Image', type: 'Image'},
            {id: '1', name: 'Text', type: 'Text'}
        ]
    }
];